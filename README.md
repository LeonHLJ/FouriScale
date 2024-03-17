## FouriScale: A Frequency Perspective on Training-Free High-Resolution Image Synthesis

<div align="center">
[Paper]() 

[Linjiang Huang](https://leonhlj.github.io/)<sup>1,2\*</sup>, [Rongyao Fang](https://scholar.google.com/citations?user=FtH3CW4AAAAJ&hl=zh-CN&oi=ao)<sup>1,\*</sup>, [Aiping Zhang]()<sup>3</sup>, [Guanglu Song]()<sup>4</sup>, [Si Liu]()<sup>5</sup>, [Yu Liu]()<sup>4</sup>, [Hongsheng Li](https://www.ee.cuhk.edu.hk/~hsli/)<sup>1,2</sup>

<sup>1</sup>CUHK-SenseTime Joint Laboratory, The Chinese University of Hong Kong<br><sup>2</sup>Centre for Perceptual and Interactive Intelligence<br><sup>3</sup>Sun Yat-Sen University, <sup>4</sup>Sensetime Research, <sup>5</sup>Beihang University
</div>

:star: If FouriScale is helpful for you, please help star this repo. Thanks!:hugs:

## :book: Table Of Contents

- [Abstract](#abstract)
- [Visual Results](#visual_results)
- [Update](#update)
- [TODO](#todo)
<!-- - [Installation](#installation)
- [Inference](#inference) -->

## <a name="abstract"></a>:fireworks: Abstract
> In this study, we delve into the generation of high-resolution images from pre-trained diffusion models, addressing persistent challenges, such as repetitive patterns and structural distortions, that emerge when models are applied beyond their trained resolutions. To address this issue, we introduce an innovative, training-free approach FouriScale from the perspective of frequency domain analysis.
We replace the original convolutional layers in pre-trained diffusion models by incorporating a dilation technique along with a low-pass operation, intending to achieve structural consistency and scale consistency across resolutions, respectively. Further enhanced by a padding-then-crop strategy, our method can flexibly handle text-to-image generation of various aspect ratios. By using the FouriScale as guidance, our method successfully balances the structural integrity and fidelity of generated images, achieving an astonishing capacity of arbitrary-size, high-resolution, and high-quality generation. With its simplicity and compatibility, our method can provide valuable insights for future explorations into the synthesis of ultra-high-resolution images.


## <a name="visual_results"></a>:eyes: Visual Results

<!-- <details close>
<summary>General Image Restoration</summary> -->
### Visual Comparisons

<img src=assets/visualization_main.jpg>

:star: Visual comparisons between ours, [ScaleCrafter](https://github.com/YingqingHe/ScaleCrafter) and [Attn-Entro](https://arxiv.org/pdf/2306.08645.pdf), under settings of 4$\times$, 8$\times$, and 16$\times$, employing three distinct pre-trained diffusion models: [SD 1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5), [SD 2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1-base), and [SDXL 1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0).

### Visual Results with LoRAs

<img src=assets/LoRA.jpg>

:star: Visualization of the high-resolution images generated by [SD 2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1-base) integrated with customized LoRAs (images in red rectangle) and images generated by a personalized diffusion model, [AnimeArtXL](https://civitai.com/models/117259/anime-art-diffusion-xl).


<!-- </details> -->

## <a name="update"></a>:new: Update

- **2024.03.18**: This repo is released :fire:
<!-- - [**History Updates** >]() -->


## <a name="todo"></a>:hourglass: TODO

- [ ] Release code :computer:
- [ ] Update links to project page :link:
- [ ] Provide HuggingFace demo :tv:


## :smiley: Citation

Please cite us if our work is useful for your research.

```
@article{2024fouriscale,
  author    = {Linjiang Huang, Rongyao Fang, Aiping Zhang, Guanglu Song, Si Liu, Yu Liu, Hongsheng Li},
  title     = {FouriScale: A Frequency Perspective on Training-Free High-Resolution Image Synthesis},
  journal   = {arxiv},
  year      = {2024},
}
```

## :notebook: License

This project is released under the [Apache 2.0 license](LICENSE).

## :bulb: Acknowledgement

Thanks [ScaleCrafter](https://github.com/YingqingHe/ScaleCrafter) for their awesome work and source code.

## :envelope: Contact

If you have any questions, please feel free to contact with me at ljhuang524@gmail.com.

