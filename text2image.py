import argparse
import math
import os
from typing import Optional

import util.misc as misc

import torch
import torch.nn.functional as F
import glob
import torch.utils.checkpoint
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL, UNet2DConditionModel, DDIMScheduler, StableDiffusionPipeline
)
from diffusers.models.attention import Attention
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from sync_tiled_decode import apply_sync_tiled_decode, apply_tiled_processors
from model import FouriConvProcessor
from free_lunch_utils import register_free_upblock2d, register_free_crossattn_upblock2d
from aux import list_layers

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--validation_prompt", type=str,
        default="a professional photograph of an astronaut riding a horse",
        help="A prompt that is sampled during training for inference."
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=23, help="A seed for reproducible training.")
    parser.add_argument("--config", type=str, default="./configs/sd1.5_1024x1024_backup.txt")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default='fp16',
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--disable_freeu", action="store_true", help="disable freeU", default=False)
    parser.add_argument("--vae_tiling", action="store_true", help="enable vae tiling")

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    args = parser.parse_args()
    return args


class TrainingFreeAttnProcessor:
    def __init__(self, name: str = None):
        self.name = name
        self.is_mid = None
        if name is not None:
            self.is_mid = "mid_block" in name

    def __call__(
            self,
            attn: Attention,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
    ):
        residual = hidden_states
        is_selfatten = encoder_hidden_states is None
        is_selfatten2 = "attn1" in self.name
        assert is_selfatten == is_selfatten2

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if is_selfatten:
            batch_size_prompt = batch_size // 3
            key = key.view(3, batch_size_prompt, attn.heads, -1, head_dim)
            query = query.view(3, batch_size_prompt, attn.heads, -1, head_dim)
            key[-1] = key[-2]
            query[-1] = query[-2]
            key = key.view(batch_size, attn.heads, -1, head_dim)
            query = query.view(batch_size, attn.heads, -1, head_dim)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def pipeline_processor(
        self,
        dilation=1.0,
        start_step=0,
        stop_step=50,
        layer_settings=None,
        base_settings=None,
        progressive=False,
):
    @torch.no_grad()
    def forward(
            prompt=None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt=None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 1.0,
            generator=None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback=None,
            callback_steps: int = 1,
            cross_attention_kwargs=None,
            guidance_rescale: float = 0.0,
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        prompt_embeds_neg, prompt_embeds_pos = prompt_embeds.chunk(2)
        prompt_embeds = torch.cat([prompt_embeds_neg, prompt_embeds_pos, prompt_embeds_pos], 0)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 3) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                unet = self.unet
                backup_forwards = dict()
                for name, module in unet.named_modules():
                    if name in layer_settings:
                        backup_forwards[name] = module.forward
                        h_base, w_base = base_settings[name]

                        if progressive and i >= start_step:
                            cur_dilation = max((dilation - 1.0) * ((stop_step - i) / (stop_step - start_step)) + 1.0, 1.0)
                        else:
                            cur_dilation = dilation

                        module.forward = FouriConvProcessor(
                            module, dilation=cur_dilation, h_base=h_base, w_base=w_base, activate=i < stop_step,
                            apply_filter=("upsamplers" not in name),
                        )

                # predict the noise residual
                noise_pred = unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                for name, module in unet.named_modules():
                    if name in backup_forwards.keys():
                        module.forward = backup_forwards[name]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, _, noise_pred_text = noise_pred.chunk(3)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return image, has_nsfw_concept

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

    return forward


def read_layer_settings(path):
    print(f"Reading layer settings")
    layer_settings = []
    with open(path, 'r') as f:
        raw_lines = f.readlines()
        for raw_line in raw_lines:
            layer_settings.append(raw_line.rstrip('\n'))
    return layer_settings


def read_base_settings(path):
    print(f"Reading base settings")
    base_settings = dict()
    with open(path, 'r') as f:
        raw_lines = f.readlines()
        for raw_line in raw_lines:
            name, dilate = raw_line.split(':')
            base_settings[name] = [float(s) for s in dilate.split(',')]
    return base_settings


def main():
    args = parse_args()

    misc.init_distributed_mode(args)
    logging_dir = os.path.join(args.logging_dir)
    config = OmegaConf.load(args.config)

    accelerator_project_config = ProjectConfiguration(logging_dir=logging_dir)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Final inference
    # Load previous pipeline
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision, torch_dtype=weight_dtype
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, torch_dtype=weight_dtype
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, torch_dtype=weight_dtype
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, torch_dtype=weight_dtype
    )
    unet.set_attn_processor({name: TrainingFreeAttnProcessor(name) for name in list_layers})
    noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    pipeline = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=noise_scheduler,
        feature_extractor=None,
        safety_checker=None
    )
    pipeline = pipeline.to(accelerator.device)
    if not args.disable_freeu:
        if 'sd1.5' in os.path.basename(args.config):
            print("Base model: SD 1.5")
            register_free_upblock2d(pipeline, b1=1.2, b2=1.4, s1=0.9, s2=0.2)
            register_free_crossattn_upblock2d(pipeline, b1=1.2, b2=1.4, s1=0.9, s2=0.2)
        elif 'sd2.1' in os.path.basename(args.config):
            print("Base model: SD 2.1")
            register_free_upblock2d(pipeline, b1=1.1, b2=1.2, s1=0.9, s2=0.2)
            register_free_crossattn_upblock2d(pipeline, b1=1.1, b2=1.2, s1=0.9, s2=0.2)
    
    if args.vae_tiling:
        pipeline.enable_vae_tiling()
        apply_sync_tiled_decode(pipeline.vae)
        apply_tiled_processors(pipeline.vae.decoder)

    layer_settings = read_layer_settings(config.layer_settings) \
        if config.layer_settings is not None else dict()

    base_settings = read_base_settings(config.base_settings) \
        if config.base_settings is not None else dict()

    unet.eval()
    os.makedirs(os.path.join(logging_dir), exist_ok=True)
    total_num = len(glob.glob(os.path.join(logging_dir, '*.jpg'))) - 1
    
    print(f"Using prompt {args.validation_prompt}")
    if os.path.isfile(args.validation_prompt):
        with open(args.validation_prompt, 'r') as f:
            validation_prompt = f.readlines()
            validation_prompt = [line.strip() for line in validation_prompt]
    else:
        validation_prompt = [args.validation_prompt, ]

    inference_batch_size = config.inference_batch_size
    num_batches = math.ceil(len(validation_prompt) / inference_batch_size)

    for i in range(num_batches):
        output_prompts = validation_prompt[i * inference_batch_size:min(
            (i + 1) * inference_batch_size, len(validation_prompt))]

        for n in range(config.num_iters_per_prompt):
            seed = args.seed + n
            set_seed(seed)

            latents = torch.randn((len(output_prompts), 4, config.latent_height, config.latent_width),
                                  device=accelerator.device, dtype=weight_dtype)

            dilation = max(math.ceil(config.latent_height / config.base_height),
                           math.ceil(config.latent_width / config.base_width))

            pipeline.forward = pipeline_processor(
                pipeline,
                dilation=dilation,
                start_step=config.start_step,
                stop_step=config.stop_step,
                layer_settings=layer_settings,
                base_settings=base_settings,
                progressive=config.progressive,
            )
            images = pipeline.forward(
                output_prompts, num_inference_steps=config.num_inference_steps, generator=None, latents=latents).images

            for image, prompt in zip(images, output_prompts):
                total_num = total_num + 1
                img_path = os.path.join(logging_dir, f"{total_num}_{prompt[:150]}_seed{seed}.jpg")
                image.save(img_path)
                with open(os.path.join(logging_dir, f"{total_num}.txt"), 'w') as f:
                    f.writelines([prompt, ])


if __name__ == "__main__":
    main()
