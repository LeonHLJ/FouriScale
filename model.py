import math
import torch
import torch.nn.functional as F

from torch import Tensor


class FouriConvProcessor:
    def __init__(self, module, dilation, h_base=1.0, w_base=1.0, apply_filter=False, activate=False,
                 weak_mask_value=0., mask_value=0.):
        self.dilation = float(dilation)
        self.h_base = h_base
        self.w_base = w_base

        self.module = module
        self.apply_filter = apply_filter
        self.activate = activate
        self.weak_mask_value = weak_mask_value
        self.mask_value = mask_value

    def create_lowpass_filter(self, input, h_scale, w_scale, value):
        B, C, H, W = input.shape
        H, R_h = H // 2, max(H // 8, 1)
        W, R_w = W // 2, max(W // 8, 1)

        if R_h == 0:
            mask_h = torch.ones(H)
        else:
            mask_h = torch.min(
                torch.max(
                    (1 - value) / R_h * (H / h_scale + 1 - torch.arange(H).float()) + 1,
                    torch.ones(H) * value
                ),
                torch.ones(H)
            )

        if R_w == 0:
            mask_w = torch.ones(W)
        else:
            mask_w = torch.min(
                torch.max(
                    (1 - value) / R_w * (W / w_scale + 1 - torch.arange(W).float()) + 1,
                    torch.ones(W) * value
                ),
                torch.ones(W)
            )

        mask = torch.ger(mask_h, mask_w)  # Use torch.ger to compute the outer product
        mask = torch.cat((torch.flip(mask, [1]), mask), 1)
        mask = torch.cat((torch.flip(mask, [0]), mask), 0)
        mask = mask.unsqueeze(0).unsqueeze(0).repeat(B // 3, C, 1, 1).to(input.device)
        return mask

    # Redefine the function with the corrected logic
    def create_weak_lowpass_filter(self, input, h_scale, w_scale, value):
        B, C, H, W = input.shape

        H, R_h = H // 2, (h_scale - 1) * H // (h_scale * 2)
        W, R_w = W // 2, (w_scale - 1) * W // (w_scale * 2)

        if R_h == 0:
            mask_h = torch.ones(H)
        else:
            mask_h = torch.min(
                torch.max(
                    (1 - value) / R_h * (H / h_scale + 1 - torch.arange(H).float()) + 1,
                    torch.ones(H) * value
                ),
                torch.ones(H)
            )

        if R_w == 0:
            mask_w = torch.ones(W)
        else:
            mask_w = torch.min(
                torch.max(
                    (1 - value) / R_w * (W / w_scale + 1 - torch.arange(W).float()) + 1,
                    torch.ones(W) * value
                ),
                torch.ones(W)
            )

        mask = torch.ger(mask_h, mask_w)  # Use torch.ger to compute the outer product
        mask = torch.cat((torch.flip(mask, [1]), mask), 1)
        mask = torch.cat((torch.flip(mask, [0]), mask), 0)
        mask = mask.unsqueeze(0).unsqueeze(0).repeat(B // 3, C, 1, 1).to(input.device)
        return mask

    def __call__(self, input: Tensor, **kwargs) -> Tensor:
        if self.activate:
            if self.dilation == 1:
                output = self.module._conv_forward(input, self.module.weight, self.module.bias)
            else:
                ori_dilation, ori_padding = self.module.dilation, self.module.padding
                self.module.dilation = (int(math.ceil(self.dilation)), int(math.ceil(self.dilation)))
                self.module.padding = (int(self.module.dilation[-2] * (self.module.weight.shape[-2] - 1) // 2),
                                       int(self.module.dilation[-1] * (self.module.weight.shape[-1] - 1) // 2))

                if self.apply_filter:
                    dtype = input.dtype
                    filtered_input = input
                    filtered_input = filtered_input.type(torch.float32)

                    r = int(max(math.ceil(filtered_input.shape[-2] / self.h_base),
                        math.ceil(filtered_input.shape[-1] / self.w_base)))
                    padding_size = (int(self.h_base * r), int(self.w_base * r))

                    if (filtered_input.shape[-2], filtered_input.shape[-1]) == padding_size:
                        pad_flag = False
                    else:
                        padding_height = padding_size[0] - filtered_input.shape[-2]
                        padding_width = padding_size[1] - filtered_input.shape[-1]
                        filtered_input = F.pad(filtered_input, (0, padding_width, 0, padding_height))
                        pad_flag = True

                    filtered_input_fft = torch.fft.fftn(filtered_input, dim=(-2, -1))
                    filtered_input_fft = torch.fft.fftshift(filtered_input_fft, dim=(-2, -1))
                    mask = self.create_lowpass_filter(filtered_input, max(self.dilation / self.module.stride[0], 1),
                                                      max(self.dilation / self.module.stride[1], 1), value=self.mask_value)
                    weak_mask = self.create_weak_lowpass_filter(filtered_input, max(self.dilation / self.module.stride[0], 1),
                        max(self.dilation / self.module.stride[1], 1), value=self.weak_mask_value)

                    low_filtered_input_fft = filtered_input_fft * torch.cat([mask, mask, weak_mask], 0)
                    low_filtered_input_fft = torch.fft.ifftshift(low_filtered_input_fft, dim=(-2, -1))
                    low_filtered_input_fft = torch.fft.ifftn(low_filtered_input_fft, dim=(-2, -1)).real

                    if pad_flag:
                        low_filtered_input_fft = low_filtered_input_fft[:, :, :low_filtered_input_fft.shape[-2]-padding_height, :low_filtered_input_fft.shape[-1]-padding_width]
                    input = low_filtered_input_fft.type(dtype)

                # forward
                output = self.module._conv_forward(input, self.module.weight, self.module.bias)
                self.module.dilation, self.module.padding = ori_dilation, ori_padding

            return output
        else:
            return self.module._conv_forward(input, self.module.weight, self.module.bias)


class FouriConvProcessor_XL:
    def __init__(self, module, dilation, target_size, base_size, value=0.6, apply_filter=False, activate=False):
        self.dilation = float(dilation)

        self.h_target = target_size[0]
        self.w_target = target_size[1]

        self.h_base = base_size[0]
        self.w_base = base_size[1]

        self.value = value

        self.module = module
        self.apply_filter = apply_filter
        self.activate = activate

    def create_lowpass_filter(self, input, h_scale, w_scale, value):
        B, C, H, W = input.shape
        H, R_h = H // 2, max(H // 8, 1)
        W, R_w = W // 2, max(W // 8, 1)

        if R_h == 0:
            mask_h = torch.ones(H)
        else:
            mask_h = torch.min(
                torch.max(
                    (1 - value) / R_h * (H / h_scale + 1 - torch.arange(H).float()) + 1,
                    torch.ones(H) * value
                ),
                torch.ones(H)
            )

        if R_w == 0:
            mask_w = torch.ones(W)
        else:
            mask_w = torch.min(
                torch.max(
                    (1 - value) / R_w * (W / w_scale + 1 - torch.arange(W).float()) + 1,
                    torch.ones(W) * value
                ),
                torch.ones(W)
            )

        mask = torch.ger(mask_h, mask_w)  # Use torch.ger to compute the outer product
        mask = torch.cat((torch.flip(mask, [1]), mask), 1)
        mask = torch.cat((torch.flip(mask, [0]), mask), 0)
        mask = mask.unsqueeze(0).unsqueeze(0).repeat(B // 3, C, 1, 1).to(input.device)
        return mask

    # Redefine the function with the corrected logic
    def create_weak_lowpass_filter(self, input, h_scale, w_scale, value):
        B, C, H, W = input.shape
        H, R_h = H // 2, (h_scale - 1) * H // (h_scale * 2)
        W, R_w = W // 2, (w_scale - 1) * W // (w_scale * 2)

        if R_h == 0:
            mask_h = torch.ones(H)
        else:
            mask_h = torch.min(
                torch.max(
                    (1 - value) / R_h * (H / h_scale + 1 - torch.arange(H).float()) + 1,
                    torch.ones(H) * value
                ),
                torch.ones(H)
            )

        if R_w == 0:
            mask_w = torch.ones(W)
        else:
            mask_w = torch.min(
                torch.max(
                    (1 - value) / R_w * (W / w_scale + 1 - torch.arange(W).float()) + 1,
                    torch.ones(W) * value
                ),
                torch.ones(W)
            )

        mask = torch.ger(mask_h, mask_w)  # Use torch.ger to compute the outer product
        mask = torch.cat((torch.flip(mask, [1]), mask), 1)
        mask = torch.cat((torch.flip(mask, [0]), mask), 0)
        mask = mask.unsqueeze(0).unsqueeze(0).repeat(B // 3, C, 1, 1).to(input.device)
        return mask

    def __call__(self, input: Tensor, **kwargs) -> Tensor:
        if self.activate:
            if self.dilation == 1:
                output = self.module._conv_forward(input, self.module.weight, self.module.bias)
            else:
                ori_dilation, ori_padding = self.module.dilation, self.module.padding
                self.module.dilation = (int(math.ceil(self.dilation)), int(math.ceil(self.dilation)))
                self.module.padding = (int(self.module.dilation[-2] * (self.module.weight.shape[-2] - 1) // 2),
                                       int(self.module.dilation[-1] * (self.module.weight.shape[-1] - 1) // 2))

                if self.apply_filter:
                    dtype = input.dtype
                    filtered_input = input.type(torch.float32)

                    down_ratio = self.h_target / filtered_input.shape[-2]
                    h_base = self.h_base / down_ratio
                    w_base = self.w_base / down_ratio

                    r = int(max(math.ceil(filtered_input.shape[-2] / h_base),
                        math.ceil(filtered_input.shape[-1] / w_base)))
                    padding_size = (int(h_base * r), int(w_base * r))

                    if (filtered_input.shape[-2], filtered_input.shape[-1]) == padding_size:
                        pad_flag = False
                    else:
                        padding_height = padding_size[0] - filtered_input.shape[-2]
                        padding_width = padding_size[1] - filtered_input.shape[-1]
                        filtered_input = F.pad(filtered_input, (0, padding_width, 0, padding_height))
                        pad_flag = True

                    filtered_input_fft = torch.fft.fftn(filtered_input, dim=(-2, -1))
                    filtered_input_fft = torch.fft.fftshift(filtered_input_fft, dim=(-2, -1))

                    mask = self.create_lowpass_filter(filtered_input, max(self.dilation / self.module.stride[0], 1),
                                                      max(self.dilation / self.module.stride[1], 1), value=self.value)
                    weak_mask = self.create_weak_lowpass_filter(filtered_input, self.dilation / self.module.stride[0],
                        self.dilation / self.module.stride[1], value=self.value)

                    low_filtered_input_fft = filtered_input_fft * torch.cat([mask, mask, weak_mask], 0)
                    low_filtered_input_fft = torch.fft.ifftshift(low_filtered_input_fft, dim=(-2, -1))
                    low_filtered_input_fft = torch.fft.ifftn(low_filtered_input_fft, dim=(-2, -1)).real

                    if pad_flag:
                        low_filtered_input_fft = low_filtered_input_fft[:, :, :low_filtered_input_fft.shape[-2]-padding_height, :low_filtered_input_fft.shape[-1]-padding_width]
                    input = low_filtered_input_fft.type(dtype)

                # forward
                B = len(input)
                output = self.module._conv_forward(input[:B // 3 * 2], self.module.weight, self.module.bias)

                # weak mask filter use smaller dilation
                self.module.dilation = (int(math.ceil(max(self.dilation / 1.5, 1))), int(math.ceil(max(self.dilation / 1.5, 1))))
                self.module.padding = (int(self.module.dilation[-2] * (self.module.weight.shape[-2] - 1) // 2),
                                       int(self.module.dilation[-1] * (self.module.weight.shape[-1] - 1) // 2))
                weak_output = self.module._conv_forward(input[-B // 3:], self.module.weight, self.module.bias)
                output = torch.cat([output, weak_output], 0)

                self.module.dilation, self.module.padding = ori_dilation, ori_padding

            return output
        else:
            return self.module._conv_forward(input, self.module.weight, self.module.bias)