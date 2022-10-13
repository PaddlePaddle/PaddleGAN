#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import random
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np

from ppgan.models.discriminators.builder import DISCRIMINATORS
from ppgan.models.generators.builder import GENERATORS
from ppgan.utils.download import get_path_from_url


class StyleGAN2Generator(nn.Layer):
    """StyleGAN2 Generator.

    Args:
        out_size (int): The spatial size of outputs.
        num_style_feat (int): Channel number of style features. Default: 512.
        num_mlp (int): Layer number of MLP style layers. Default: 8.
        channel_multiplier (int): Channel multiplier for large networks of
            StyleGAN2. Default: 2.
        resample_kernel (list[int]): A list indicating the 1D resample kernel
            magnitude. A cross production will be applied to extent 1D resample
            kenrel to 2D resample kernel. Default: (1, 3, 3, 1).
        lr_mlp (float): Learning rate multiplier for mlp layers. Default: 0.01.
        narrow (float): Narrow ratio for channels. Default: 1.0.
    """
    def __init__(self,
                 out_size,
                 num_style_feat=512,
                 num_mlp=8,
                 channel_multiplier=2,
                 resample_kernel=(1, 3, 3, 1),
                 lr_mlp=0.01,
                 narrow=1):
        super(StyleGAN2Generator, self).__init__()
        self.num_style_feat = num_style_feat
        style_mlp_layers = [NormStyleCode()]
        for i in range(num_mlp):
            style_mlp_layers.append(
                EqualLinear(num_style_feat,
                            num_style_feat,
                            bias=True,
                            bias_init_val=0,
                            lr_mul=lr_mlp,
                            activation='fused_lrelu'))
        self.style_mlp = nn.Sequential(*style_mlp_layers)
        channels = {
            '4': int(512 * narrow),
            '8': int(512 * narrow),
            '16': int(512 * narrow),
            '32': int(512 * narrow),
            '64': int(256 * channel_multiplier * narrow),
            '128': int(128 * channel_multiplier * narrow),
            '256': int(64 * channel_multiplier * narrow),
            '512': int(32 * channel_multiplier * narrow),
            '1024': int(16 * channel_multiplier * narrow)
        }
        self.channels = channels
        self.constant_input = ConstantInput(channels['4'], size=4)
        self.style_conv1 = StyleConv(channels['4'],
                                     channels['4'],
                                     kernel_size=3,
                                     num_style_feat=num_style_feat,
                                     demodulate=True,
                                     sample_mode=None,
                                     resample_kernel=resample_kernel)
        self.to_rgb1 = ToRGB(channels['4'],
                             num_style_feat,
                             upsample=False,
                             resample_kernel=resample_kernel)
        self.log_size = int(math.log(out_size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1
        self.num_latent = self.log_size * 2 - 2
        self.style_convs = nn.LayerList()
        self.to_rgbs = nn.LayerList()
        self.noises = nn.Layer()
        in_channels = channels['4']
        for layer_idx in range(self.num_layers):
            resolution = 2**((layer_idx + 5) // 2)
            shape = [1, 1, resolution, resolution]
            x = paddle.ones(shape=shape, dtype='float32')
            self.noises.register_buffer(f'noise{layer_idx}', x)
        for i in range(3, self.log_size + 1):
            out_channels = channels[f'{2 ** i}']
            self.style_convs.append(
                StyleConv(in_channels,
                          out_channels,
                          kernel_size=3,
                          num_style_feat=num_style_feat,
                          demodulate=True,
                          sample_mode='upsample',
                          resample_kernel=resample_kernel))
            self.style_convs.append(
                StyleConv(out_channels,
                          out_channels,
                          kernel_size=3,
                          num_style_feat=num_style_feat,
                          demodulate=True,
                          sample_mode=None,
                          resample_kernel=resample_kernel))
            self.to_rgbs.append(
                ToRGB(out_channels,
                      num_style_feat,
                      upsample=True,
                      resample_kernel=resample_kernel))
            in_channels = out_channels

    def make_noise(self):
        """Make noise for noise injection."""
        device = self.constant_input.weight.device
        x = paddle.ones(shape=[1, 1, 4, 4], dtype='float32')
        noises = [x]
        for i in range(3, self.log_size + 1):
            for _ in range(2):
                x = paddle.ones(shape=[1, 1, 2**i, 2**i], dtype='float32')
                noises.append(x)
        return noises

    def get_latent(self, x):
        return self.style_mlp(x)

    def mean_latent(self, num_latent):
        x = paddle.ones(shape=[num_latent, self.num_style_feat],
                        dtype='float32')
        latent_in = x
        latent = self.style_mlp(latent_in).mean(0, keepdim=True)
        return latent

    def forward(self,
                styles,
                input_is_latent=False,
                noise=None,
                randomize_noise=True,
                truncation=1,
                truncation_latent=None,
                inject_index=None,
                return_latents=False):
        """Forward function for StyleGAN2Generator.

        Args:
            styles (list[Tensor]): Sample codes of styles.
            input_is_latent (bool): Whether input is latent style.
                Default: False.
            noise (Tensor | None): Input noise or None. Default: None.
            randomize_noise (bool): Randomize noise, used when 'noise' is
                False. Default: True.
            truncation (float): TODO. Default: 1.
            truncation_latent (Tensor | None): TODO. Default: None.
            inject_index (int | None): The injection index for mixing noise.
                Default: None.
            return_latents (bool): Whether to return style latents.
                Default: False.
        """
        if not input_is_latent:
            styles = [self.style_mlp(s) for s in styles]
        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f'noise{i}')
                    for i in range(self.num_layers)
                ]
        if truncation < 1:
            style_truncation = []
            for style in styles:
                style_truncation.append(truncation_latent + truncation *
                                        (style - truncation_latent))
            styles = style_truncation
        if len(styles) == 1:
            inject_index = self.num_latent
            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1)
                latent = paddle.tile(latent, repeat_times=[1, inject_index, 1])
            else:
                latent = styles[0]
        elif len(styles) == 2:
            if inject_index is None:
                inject_index = random.randint(1, self.num_latent - 1)
            latent1 = styles[0].unsqueeze(1)
            latent1 = paddle.tile(latent, repeat_times=[1, inject_index, 1])

            latent2 = styles[1].unsqueeze(1)
            latent2 = paddle.tile(
                latent2, repeat_times=[1, self.num_latent - inject_index, 1])
            latent = paddle.concat([latent1, latent2], 1)
        out = self.constant_input(latent.shape[0])
        out = self.style_conv1(out, latent[:, 0], noise=noise[0])
        skip = self.to_rgb1(out, latent[:, 1])
        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(self.style_convs[::2],
                                                        self.style_convs[1::2],
                                                        noise[1::2],
                                                        noise[2::2],
                                                        self.to_rgbs):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)
            i += 2
        image = skip
        if return_latents:
            return image, latent
        else:
            return image, None


def var(x, axis=None, unbiased=True, keepdim=False, name=None):

    u = paddle.mean(x, axis, True, name)
    out = paddle.sum((x - u) * (x - u), axis, keepdim=keepdim, name=name)

    n = paddle.cast(paddle.numel(x), x.dtype) \
        / paddle.cast(paddle.numel(out), x.dtype)
    if unbiased:
        one_const = paddle.ones([1], x.dtype)
        n = paddle.where(n > one_const, n - 1., one_const)
    out /= n
    return out


@DISCRIMINATORS.register()
class StyleGAN2DiscriminatorGFPGAN(nn.Layer):
    """StyleGAN2 Discriminator.

    Args:
        out_size (int): The spatial size of outputs.
        channel_multiplier (int): Channel multiplier for large networks of
            StyleGAN2. Default: 2.
        resample_kernel (list[int]): A list indicating the 1D resample kernel
            magnitude. A cross production will be applied to extent 1D resample
            kenrel to 2D resample kernel. Default: (1, 3, 3, 1).
        stddev_group (int): For group stddev statistics. Default: 4.
        narrow (float): Narrow ratio for channels. Default: 1.0.
    """
    def __init__(self,
                 out_size,
                 channel_multiplier=2,
                 resample_kernel=(1, 3, 3, 1),
                 stddev_group=4,
                 narrow=1):
        super(StyleGAN2DiscriminatorGFPGAN, self).__init__()
        channels = {
            '4': int(512 * narrow),
            '8': int(512 * narrow),
            '16': int(512 * narrow),
            '32': int(512 * narrow),
            '64': int(256 * channel_multiplier * narrow),
            '128': int(128 * channel_multiplier * narrow),
            '256': int(64 * channel_multiplier * narrow),
            '512': int(32 * channel_multiplier * narrow),
            '1024': int(16 * channel_multiplier * narrow)
        }
        log_size = int(math.log(out_size, 2))
        conv_body = [
            ConvLayer(3, channels[f'{out_size}'], 1, bias=True, activate=True)
        ]
        in_channels = channels[f'{out_size}']
        for i in range(log_size, 2, -1):
            out_channels = channels[f'{2 ** (i - 1)}']
            conv_body.append(
                ResBlock(in_channels, out_channels, resample_kernel))
            in_channels = out_channels
        self.conv_body = nn.Sequential(*conv_body)
        self.final_conv = ConvLayer(in_channels + 1,
                                    channels['4'],
                                    3,
                                    bias=True,
                                    activate=True)
        self.final_linear = nn.Sequential(
            EqualLinear(channels['4'] * 4 * 4,
                        channels['4'],
                        bias=True,
                        bias_init_val=0,
                        lr_mul=1,
                        activation='fused_lrelu'),
            EqualLinear(channels['4'],
                        1,
                        bias=True,
                        bias_init_val=0,
                        lr_mul=1,
                        activation=None))
        self.stddev_group = stddev_group
        self.stddev_feat = 1

    def forward(self, x):
        out = self.conv_body(x)
        b, c, h, w = out.shape
        group = min(b, self.stddev_group)
        stddev = out.reshape(
            [group, -1, self.stddev_feat, c // self.stddev_feat, h, w])
        stddev = paddle.sqrt(var(stddev, 0, unbiased=False) + 1e-08)
        stddev = stddev.mean(axis=[2, 3, 4], keepdim=True).squeeze(2)

        stddev = paddle.tile(stddev, repeat_times=[group, 1, h, w])
        out = paddle.concat([out, stddev], 1)
        out = self.final_conv(out)
        out = out.reshape([b, -1])
        out = self.final_linear(out)
        return out


class StyleGAN2GeneratorSFT(StyleGAN2Generator):
    """StyleGAN2 Generator with SFT modulation (Spatial Feature Transform).

    Args:
        out_size (int): The spatial size of outputs.
        num_style_feat (int): Channel number of style features. Default: 512.
        num_mlp (int): Layer number of MLP style layers. Default: 8.
        channel_multiplier (int): Channel multiplier for large networks of StyleGAN2. Default: 2.
        resample_kernel (list[int]): A list indicating the 1D resample kernel magnitude. A cross production will be
            applied to extent 1D resample kernel to 2D resample kernel. Default: (1, 3, 3, 1).
        lr_mlp (float): Learning rate multiplier for mlp layers. Default: 0.01.
        narrow (float): The narrow ratio for channels. Default: 1.
        sft_half (bool): Whether to apply SFT on half of the input channels. Default: False.
    """
    def __init__(self,
                 out_size,
                 num_style_feat=512,
                 num_mlp=8,
                 channel_multiplier=2,
                 resample_kernel=(1, 3, 3, 1),
                 lr_mlp=0.01,
                 narrow=1,
                 sft_half=False):
        super(StyleGAN2GeneratorSFT,
              self).__init__(out_size,
                             num_style_feat=num_style_feat,
                             num_mlp=num_mlp,
                             channel_multiplier=channel_multiplier,
                             resample_kernel=resample_kernel,
                             lr_mlp=lr_mlp,
                             narrow=narrow)
        self.sft_half = sft_half

    def forward(self,
                styles,
                conditions,
                input_is_latent=False,
                noise=None,
                randomize_noise=True,
                truncation=1,
                truncation_latent=None,
                inject_index=None,
                return_latents=False):
        """Forward function for StyleGAN2GeneratorSFT.

        Args:
            styles (list[Tensor]): Sample codes of styles.
            conditions (list[Tensor]): SFT conditions to generators.
            input_is_latent (bool): Whether input is latent style. Default: False.
            noise (Tensor | None): Input noise or None. Default: None.
            randomize_noise (bool): Randomize noise, used when 'noise' is False. Default: True.
            truncation (float): The truncation ratio. Default: 1.
            truncation_latent (Tensor | None): The truncation latent tensor. Default: None.
            inject_index (int | None): The injection index for mixing noise. Default: None.
            return_latents (bool): Whether to return style latents. Default: False.
        """
        if not input_is_latent:
            styles = [self.style_mlp(s) for s in styles]
        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f'noise{i}')
                    for i in range(self.num_layers)
                ]
        if truncation < 1:
            style_truncation = []
            for style in styles:
                style_truncation.append(truncation_latent + truncation *
                                        (style - truncation_latent))
            styles = style_truncation
        if len(styles) == 1:
            inject_index = self.num_latent
            if styles[0].ndim < 3:
                latent = paddle.tile(styles[0].unsqueeze(1),
                                     repeat_times=[1, inject_index, 1])
            else:
                latent = styles[0]
        elif len(styles) == 2:
            if inject_index is None:
                inject_index = random.randint(1, self.num_latent - 1)
            latent1 = styles[0].unsqueeze(1)
            latent1 = paddle.tile(latent, repeat_times=[1, inject_index, 1])

            latent2 = styles[1].unsqueeze(1)
            latent2 = paddle.tile(
                latent2, repeat_times=[1, self.num_latent - inject_index, 1])
            latent = paddle.concat([latent1, latent2], 1)
        out = self.constant_input(latent.shape[0])
        out = self.style_conv1(out, latent[:, 0], noise=noise[0])
        skip = self.to_rgb1(out, latent[:, 1])
        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(self.style_convs[::2],
                                                        self.style_convs[1::2],
                                                        noise[1::2],
                                                        noise[2::2],
                                                        self.to_rgbs):
            out = conv1(out, latent[:, i], noise=noise1)
            if i < len(conditions):
                if self.sft_half:
                    out_same, out_sft = paddle.split(out, 2, axis=1)
                    out_sft = out_sft * conditions[i - 1] + conditions[i]
                    out = paddle.concat([out_same, out_sft], axis=1)
                else:
                    out = out * conditions[i - 1] + conditions[i]
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)
            i += 2
        image = skip
        if return_latents:
            return image, latent
        else:
            return image, None


@GENERATORS.register()
class GFPGANv1(nn.Layer):
    """The GFPGAN architecture: Unet + StyleGAN2 decoder with SFT.

    Ref: GFP-GAN: Towards Real-World Blind Face Restoration with Generative Facial Prior.

    Args:
        out_size (int): The spatial size of outputs.
        num_style_feat (int): Channel number of style features. Default: 512.
        channel_multiplier (int): Channel multiplier for large networks of StyleGAN2. Default: 2.
        resample_kernel (list[int]): A list indicating the 1D resample kernel magnitude. A cross production will be
            applied to extent 1D resample kernel to 2D resample kernel. Default: (1, 3, 3, 1).
        decoder_load_path (str): The path to the pre-trained decoder model (usually, the StyleGAN2). Default: None.
        fix_decoder (bool): Whether to fix the decoder. Default: True.

        num_mlp (int): Layer number of MLP style layers. Default: 8.
        lr_mlp (float): Learning rate multiplier for mlp layers. Default: 0.01.
        input_is_latent (bool): Whether input is latent style. Default: False.
        different_w (bool): Whether to use different latent w for different layers. Default: False.
        narrow (float): The narrow ratio for channels. Default: 1.
        sft_half (bool): Whether to apply SFT on half of the input channels. Default: False.
    """
    def __init__(self,
                 out_size,
                 num_style_feat=512,
                 channel_multiplier=1,
                 resample_kernel=(1, 3, 3, 1),
                 decoder_load_path=None,
                 fix_decoder=True,
                 num_mlp=8,
                 lr_mlp=0.01,
                 input_is_latent=False,
                 different_w=False,
                 narrow=1,
                 sft_half=False):
        super(GFPGANv1, self).__init__()
        self.input_is_latent = input_is_latent
        self.different_w = different_w
        self.num_style_feat = num_style_feat
        unet_narrow = narrow * 0.5
        channels = {
            '4': int(512 * unet_narrow),
            '8': int(512 * unet_narrow),
            '16': int(512 * unet_narrow),
            '32': int(512 * unet_narrow),
            '64': int(256 * channel_multiplier * unet_narrow),
            '128': int(128 * channel_multiplier * unet_narrow),
            '256': int(64 * channel_multiplier * unet_narrow),
            '512': int(32 * channel_multiplier * unet_narrow),
            '1024': int(16 * channel_multiplier * unet_narrow)
        }
        self.log_size = int(math.log(out_size, 2))
        first_out_size = 2**int(math.log(out_size, 2))
        self.conv_body_first = ConvLayer(3,
                                         channels[f'{first_out_size}'],
                                         1,
                                         bias=True,
                                         activate=True)
        in_channels = channels[f'{first_out_size}']
        self.conv_body_down = nn.LayerList()
        for i in range(self.log_size, 2, -1):
            out_channels = channels[f'{2 ** (i - 1)}']
            self.conv_body_down.append(
                ResBlock(in_channels, out_channels, resample_kernel))
            in_channels = out_channels
        self.final_conv = ConvLayer(in_channels,
                                    channels['4'],
                                    3,
                                    bias=True,
                                    activate=True)
        in_channels = channels['4']
        self.conv_body_up = nn.LayerList()
        for i in range(3, self.log_size + 1):
            out_channels = channels[f'{2 ** i}']
            self.conv_body_up.append(ResUpBlock(in_channels, out_channels))
            in_channels = out_channels
        self.toRGB = nn.LayerList()
        for i in range(3, self.log_size + 1):
            self.toRGB.append(
                EqualConv2d(channels[f'{2 ** i}'],
                            3,
                            1,
                            stride=1,
                            padding=0,
                            bias=True,
                            bias_init_val=0))
        if different_w:
            linear_out_channel = (int(math.log(out_size, 2)) * 2 -
                                  2) * num_style_feat
        else:
            linear_out_channel = num_style_feat
        self.final_linear = EqualLinear(channels['4'] * 4 * 4,
                                        linear_out_channel,
                                        bias=True,
                                        bias_init_val=0,
                                        lr_mul=1,
                                        activation=None)
        self.stylegan_decoder = StyleGAN2GeneratorSFT(
            out_size=out_size,
            num_style_feat=num_style_feat,
            num_mlp=num_mlp,
            channel_multiplier=channel_multiplier,
            resample_kernel=resample_kernel,
            lr_mlp=lr_mlp,
            narrow=narrow,
            sft_half=sft_half)
        if decoder_load_path:
            decoder_load_path = get_path_from_url(decoder_load_path)
            self.stylegan_decoder.set_state_dict(paddle.load(decoder_load_path))

        if fix_decoder:
            for _, param in self.stylegan_decoder.named_parameters():
                param.stop_gradient = True
        self.condition_scale = nn.LayerList()
        self.condition_shift = nn.LayerList()
        for i in range(3, self.log_size + 1):
            out_channels = channels[f'{2 ** i}']
            if sft_half:
                sft_out_channels = out_channels
            else:
                sft_out_channels = out_channels * 2
            self.condition_scale.append(
                nn.Sequential(
                    EqualConv2d(out_channels,
                                out_channels,
                                3,
                                stride=1,
                                padding=1,
                                bias=True,
                                bias_init_val=0), ScaledLeakyReLU(0.2),
                    EqualConv2d(out_channels,
                                sft_out_channels,
                                3,
                                stride=1,
                                padding=1,
                                bias=True,
                                bias_init_val=1)))
            self.condition_shift.append(
                nn.Sequential(
                    EqualConv2d(out_channels,
                                out_channels,
                                3,
                                stride=1,
                                padding=1,
                                bias=True,
                                bias_init_val=0), ScaledLeakyReLU(0.2),
                    EqualConv2d(out_channels,
                                sft_out_channels,
                                3,
                                stride=1,
                                padding=1,
                                bias=True,
                                bias_init_val=0)))

    def forward(self,
                x,
                return_latents=False,
                return_rgb=True,
                randomize_noise=False):
        """Forward function for GFPGANv1.

        Args:
            x (Tensor): Input images.
            return_latents (bool): Whether to return style latents. Default: False.
            return_rgb (bool): Whether return intermediate rgb images. Default: True.
            randomize_noise (bool): Randomize noise, used when 'noise' is False. Default: True.
        """
        conditions = []
        unet_skips = []
        out_rgbs = []

        feat = self.conv_body_first(x)

        for i in range(self.log_size - 2):
            feat = self.conv_body_down[i](feat)
            unet_skips.insert(0, feat)
        feat = self.final_conv(feat)
        style_code = self.final_linear(feat.reshape([feat.shape[0], -1]))
        if self.different_w:
            style_code = style_code.reshape(
                [style_code.shape[0], -1, self.num_style_feat])

        for i in range(self.log_size - 2):
            feat = feat + unet_skips[i]
            feat = self.conv_body_up[i](feat)
            scale = self.condition_scale[i](feat)
            conditions.append(scale.clone())
            shift = self.condition_shift[i](feat)
            conditions.append(shift.clone())
            if return_rgb:
                out_rgbs.append(self.toRGB[i](feat))
        image, _ = self.stylegan_decoder([style_code],
                                         conditions,
                                         return_latents=return_latents,
                                         input_is_latent=self.input_is_latent,
                                         randomize_noise=randomize_noise)
        return image, out_rgbs


class FacialComponentDiscriminator(nn.Layer):
    """Facial component (eyes, mouth, noise) discriminator used in GFPGAN.
    """
    def __init__(self):
        super(FacialComponentDiscriminator, self).__init__()
        self.conv1 = ConvLayer(3,
                               64,
                               3,
                               downsample=False,
                               resample_kernel=(1, 3, 3, 1),
                               bias=True,
                               activate=True)
        self.conv2 = ConvLayer(64,
                               128,
                               3,
                               downsample=True,
                               resample_kernel=(1, 3, 3, 1),
                               bias=True,
                               activate=True)
        self.conv3 = ConvLayer(128,
                               128,
                               3,
                               downsample=False,
                               resample_kernel=(1, 3, 3, 1),
                               bias=True,
                               activate=True)
        self.conv4 = ConvLayer(128,
                               256,
                               3,
                               downsample=True,
                               resample_kernel=(1, 3, 3, 1),
                               bias=True,
                               activate=True)
        self.conv5 = ConvLayer(256,
                               256,
                               3,
                               downsample=False,
                               resample_kernel=(1, 3, 3, 1),
                               bias=True,
                               activate=True)
        self.final_conv = ConvLayer(256, 1, 3, bias=True, activate=False)

    def forward(self, x, return_feats=False):
        """Forward function for FacialComponentDiscriminator.

        Args:
            x (Tensor): Input images.
            return_feats (bool): Whether to return intermediate features. Default: False.
        """
        feat = self.conv1(x)
        feat = self.conv3(self.conv2(feat))
        rlt_feats = []
        if return_feats:
            rlt_feats.append(feat.clone())
        feat = self.conv5(self.conv4(feat))
        if return_feats:
            rlt_feats.append(feat.clone())
        out = self.final_conv(feat)
        if return_feats:
            return out, rlt_feats
        else:
            return out, None


class ConvUpLayer(nn.Layer):
    """Convolutional upsampling layer. It uses bilinear upsampler + Conv.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        kernel_size (int): Size of the convolving kernel.
        stride (int): Stride of the convolution. Default: 1
        padding (int): Zero-padding added to both sides of the input. Default: 0.
        bias (bool): If ``True``, adds a learnable bias to the output. Default: ``True``.
        bias_init_val (float): Bias initialized value. Default: 0.
        activate (bool): Whether use activateion. Default: True.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=True,
                 bias_init_val=0,
                 activate=True):
        super(ConvUpLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.scale = 1 / math.sqrt(in_channels * kernel_size**2)
        self.weight = paddle.create_parameter(
            shape=[out_channels, in_channels, kernel_size, kernel_size],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Normal())
        if bias and not activate:
            self.bias = paddle.create_parameter(
                shape=[out_channels],
                dtype='float32',
                default_initializer=paddle.nn.initializer.Constant(
                    bias_init_val))
        else:
            pass
            self.bias = None
        if activate:
            if bias:
                self.activation = FusedLeakyReLU(out_channels)
            else:
                self.activation = ScaledLeakyReLU(0.2)
        else:
            self.activation = None

    def forward(self, x):
        out = F.interpolate(x,
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=False)
        out = F.conv2d(out,
                       self.weight * self.scale,
                       bias=self.bias,
                       stride=self.stride,
                       padding=self.padding)
        if self.activation is not None:
            out = self.activation(out)
        return out


class ResUpBlock(nn.Layer):
    """Residual block with upsampling.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
    """
    def __init__(self, in_channels, out_channels):
        super(ResUpBlock, self).__init__()
        self.conv1 = ConvLayer(in_channels,
                               in_channels,
                               3,
                               bias=True,
                               activate=True)
        self.conv2 = ConvUpLayer(in_channels,
                                 out_channels,
                                 3,
                                 stride=1,
                                 padding=1,
                                 bias=True,
                                 activate=True)
        self.skip = ConvUpLayer(in_channels,
                                out_channels,
                                1,
                                bias=False,
                                activate=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        skip = self.skip(x)
        out = (out + skip) / math.sqrt(2)
        return out


def upfirdn2d_native(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1,
                     pad_y0, pad_y1):
    _, channel, in_h, in_w = input.shape
    input = input.reshape((-1, in_h, in_w, 1))
    _, in_h, in_w, minor = input.shape
    kernel_h, kernel_w = kernel.shape
    out = input.reshape((-1, in_h, 1, in_w, 1, minor))
    out = out.transpose((0, 1, 3, 5, 2, 4))
    out = out.reshape((-1, 1, 1, 1))
    out = F.pad(out, [0, up_x - 1, 0, up_y - 1])
    out = out.reshape((-1, in_h, in_w, minor, up_y, up_x))
    out = out.transpose((0, 3, 1, 4, 2, 5))
    out = out.reshape((-1, minor, in_h * up_y, in_w * up_x))
    out = F.pad(
        out, [max(pad_x0, 0),
              max(pad_x1, 0),
              max(pad_y0, 0),
              max(pad_y1, 0)])
    out = out[:, :,
              max(-pad_y0, 0):out.shape[2] - max(-pad_y1, 0),
              max(-pad_x0, 0):out.shape[3] - max(-pad_x1, 0)]
    out = out.reshape(
        [-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1])
    w = paddle.flip(kernel, [0, 1]).reshape((1, 1, kernel_h, kernel_w))
    out = F.conv2d(out, w)
    out = out.reshape((-1, minor, in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
                       in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1))
    out = out.transpose((0, 2, 3, 1))
    out = out[:, ::down_y, ::down_x, :]
    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1
    return out.reshape((-1, channel, out_h, out_w))


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    out = upfirdn2d_native(input, kernel, up, up, down, down, pad[0], pad[1],
                           pad[0], pad[1])
    return out


class NormStyleCode(nn.Layer):
    def forward(self, x):
        """Normalize the style codes.

        Args:
            x (Tensor): Style codes with shape (b, c).

        Returns:
            Tensor: Normalized tensor.
        """
        return x * paddle.rsqrt(paddle.mean(x**2, axis=1, keepdim=True) + 1e-08)


def make_resample_kernel(k):
    """Make resampling kernel for UpFirDn.

    Args:
        k (list[int]): A list indicating the 1D resample kernel magnitude.

    Returns:
        Tensor: 2D resampled kernel.
    """
    k = paddle.to_tensor(k, dtype="float32")
    if k.ndim == 1:
        k = k[None, :] * k[:, None]
    k /= k.sum()
    return k


class UpFirDnUpsample(nn.Layer):
    """Upsample, FIR filter, and downsample (upsampole version).

    References:
    1. https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.upfirdn.html  # noqa: E501
    2. http://www.ece.northwestern.edu/local-apps/matlabhelp/toolbox/signal/upfirdn.html  # noqa: E501

    Args:
        resample_kernel (list[int]): A list indicating the 1D resample kernel
            magnitude.
        factor (int): Upsampling scale factor. Default: 2.
    """
    def __init__(self, resample_kernel, factor=2):
        super(UpFirDnUpsample, self).__init__()
        self.kernel = make_resample_kernel(resample_kernel) * factor**2
        self.factor = factor
        pad = self.kernel.shape[0] - factor
        self.pad = (pad + 1) // 2 + factor - 1, pad // 2

    def forward(self, x):
        out = upfirdn2d(x, self.kernel, up=self.factor, down=1, pad=self.pad)
        return out

    def __repr__(self):
        return f'{self.__class__.__name__}(factor={self.factor})'


class UpFirDnDownsample(nn.Layer):
    """Upsample, FIR filter, and downsample (downsampole version).

    Args:
        resample_kernel (list[int]): A list indicating the 1D resample kernel
            magnitude.
        factor (int): Downsampling scale factor. Default: 2.
    """
    def __init__(self, resample_kernel, factor=2):
        super(UpFirDnDownsample, self).__init__()
        self.kernel = make_resample_kernel(resample_kernel)
        self.factor = factor
        pad = self.kernel.shape[0] - factor
        self.pad = (pad + 1) // 2, pad // 2

    def forward(self, x):
        out = upfirdn2d(x, self.kernel, up=1, down=self.factor, pad=self.pad)
        return out

    def __repr__(self):
        return f'{self.__class__.__name__}(factor={self.factor})'


class UpFirDnSmooth(nn.Layer):
    """Upsample, FIR filter, and downsample (smooth version).

    Args:
        resample_kernel (list[int]): A list indicating the 1D resample kernel
            magnitude.
        upsample_factor (int): Upsampling scale factor. Default: 1.
        downsample_factor (int): Downsampling scale factor. Default: 1.
        kernel_size (int): Kernel size: Deafult: 1.
    """
    def __init__(self,
                 resample_kernel,
                 upsample_factor=1,
                 downsample_factor=1,
                 kernel_size=1):
        super(UpFirDnSmooth, self).__init__()
        self.upsample_factor = upsample_factor
        self.downsample_factor = downsample_factor
        self.kernel = make_resample_kernel(resample_kernel)
        if upsample_factor > 1:
            self.kernel = self.kernel * upsample_factor**2
        if upsample_factor > 1:
            pad = self.kernel.shape[0] - upsample_factor - (kernel_size - 1)
            self.pad = (pad + 1) // 2 + upsample_factor - 1, pad // 2 + 1
        elif downsample_factor > 1:
            pad = self.kernel.shape[0] - downsample_factor + (kernel_size - 1)
            self.pad = (pad + 1) // 2, pad // 2
        else:
            raise NotImplementedError

    def forward(self, x):
        out = upfirdn2d(x, self.kernel, up=1, down=1, pad=self.pad)
        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(upsample_factor={self.upsample_factor}, \
            downsample_factor={self.downsample_factor})')


class EqualLinear(nn.Layer):
    """This linear layer class stabilizes the learning rate changes of its parameters.
    Equalizing learning rate keeps the weights in the network at a similar scale during training.
    """
    def __init__(self,
                 in_dim,
                 out_dim,
                 bias=True,
                 bias_init_val=0,
                 lr_mul=1,
                 activation=None):
        super().__init__()

        self.weight = paddle.create_parameter(
            (in_dim, out_dim),
            default_initializer=nn.initializer.Normal(),
            dtype='float32')
        self.weight.set_value((self.weight / lr_mul))

        if bias:
            self.bias = self.create_parameter(
                (out_dim, ), nn.initializer.Constant(bias_init_val))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(input,
                           self.weight * self.scale,
                           bias=self.bias * self.lr_mul)

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[0]}, {self.weight.shape[1]})"
        )


class ModulatedConv2d(nn.Layer):
    """Modulated Conv2d used in StyleGAN2.

    There is no bias in ModulatedConv2d.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        kernel_size (int): Size of the convolving kernel.
        num_style_feat (int): Channel number of style features.
        demodulate (bool): Whether to demodulate in the conv layer.
            Default: True.
        sample_mode (str | None): Indicating 'upsample', 'downsample' or None.
            Default: None.
        resample_kernel (list[int]): A list indicating the 1D resample kernel
            magnitude. Default: (1, 3, 3, 1).
        eps (float): A value added to the denominator for numerical stability.
            Default: 1e-8.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 num_style_feat,
                 demodulate=True,
                 sample_mode=None,
                 resample_kernel=(1, 3, 3, 1),
                 eps=1e-08):
        super(ModulatedConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.sample_mode = sample_mode
        self.eps = eps
        if self.sample_mode == 'upsample':
            self.smooth = UpFirDnSmooth(resample_kernel,
                                        upsample_factor=2,
                                        downsample_factor=1,
                                        kernel_size=kernel_size)
        elif self.sample_mode == 'downsample':
            self.smooth = UpFirDnSmooth(resample_kernel,
                                        upsample_factor=1,
                                        downsample_factor=2,
                                        kernel_size=kernel_size)
        elif self.sample_mode is None:
            pass
        else:
            raise ValueError(
                f"Wrong sample mode {self.sample_mode}, supported ones are ['upsample', 'downsample', None]."
            )
        self.scale = 1 / math.sqrt(in_channels * kernel_size**2)
        self.modulation = EqualLinear(num_style_feat,
                                      in_channels,
                                      bias=True,
                                      bias_init_val=1,
                                      lr_mul=1,
                                      activation=None)
        self.weight = paddle.create_parameter(
            shape=[1, out_channels, in_channels, kernel_size, kernel_size],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Normal())
        self.padding = kernel_size // 2

    def forward(self, x, style):
        """Forward function.

        Args:
            x (Tensor): Tensor with shape (b, c, h, w).
            style (Tensor): Tensor with shape (b, num_style_feat).

        Returns:
            Tensor: Modulated tensor after convolution.
        """
        b, c, h, w = x.shape
        style = self.modulation(style).reshape([b, 1, c, 1, 1])
        weight = self.scale * self.weight * style
        if self.demodulate:
            demod = paddle.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps)
            weight = weight * demod.reshape([b, self.out_channels, 1, 1, 1])
        weight = weight.reshape(
            [b * self.out_channels, c, self.kernel_size, self.kernel_size])
        if self.sample_mode == 'upsample':
            x = x.reshape([1, b * c, h, w])
            weight = weight.reshape(
                [b, self.out_channels, c, self.kernel_size, self.kernel_size])
            weight = weight.transpose([0, 2, 1, 3, 4]).reshape(
                [b * c, self.out_channels, self.kernel_size, self.kernel_size])
            out = F.conv2d_transpose(x, weight, padding=0, stride=2, groups=b)
            out = out.reshape([b, self.out_channels, *out.shape[2:4]])
            out = self.smooth(out)
        elif self.sample_mode == 'downsample':
            x = self.smooth(x)
            x = x.reshape([1, b * c, *x.shape[2:4]])
            out = F.conv2d(x, weight, padding=0, stride=2, groups=b)
            out = out.reshape([b, self.out_channels, *out.shape[2:4]])
        else:
            x = x.reshape([1, b * c, h, w])
            out = F.conv2d(x, weight, padding=self.padding, groups=b)
            out = out.reshape([b, self.out_channels, *out.shape[2:4]])
        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}(in_channels={self.in_channels}, \
            out_channels={self.out_channels}, \
            kernel_size={self.kernel_size}, \
            demodulate={self.demodulate}, \
            sample_mode={self.sample_mode})')


class StyleConv(nn.Layer):
    """Style conv.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        kernel_size (int): Size of the convolving kernel.
        num_style_feat (int): Channel number of style features.
        demodulate (bool): Whether demodulate in the conv layer. Default: True.
        sample_mode (str | None): Indicating 'upsample', 'downsample' or None.
            Default: None.
        resample_kernel (list[int]): A list indicating the 1D resample kernel
            magnitude. Default: (1, 3, 3, 1).
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 num_style_feat,
                 demodulate=True,
                 sample_mode=None,
                 resample_kernel=(1, 3, 3, 1)):
        super(StyleConv, self).__init__()
        self.modulated_conv = ModulatedConv2d(in_channels,
                                              out_channels,
                                              kernel_size,
                                              num_style_feat,
                                              demodulate=demodulate,
                                              sample_mode=sample_mode,
                                              resample_kernel=resample_kernel)
        self.weight = paddle.create_parameter(
            shape=[1],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Constant(0.))
        self.activate = FusedLeakyReLU(out_channels)

    def forward(self, x, style, noise=None):
        out = self.modulated_conv(x, style)
        if noise is None:
            b, _, h, w = out.shape
            noise = paddle.normal(shape=[b, 1, h, w])
        out = out + self.weight * noise
        out = self.activate(out)
        return out


class ToRGB(nn.Layer):
    """To RGB from features.

    Args:
        in_channels (int): Channel number of input.
        num_style_feat (int): Channel number of style features.
        upsample (bool): Whether to upsample. Default: True.
        resample_kernel (list[int]): A list indicating the 1D resample kernel
            magnitude. Default: (1, 3, 3, 1).
    """
    def __init__(self,
                 in_channels,
                 num_style_feat,
                 upsample=True,
                 resample_kernel=(1, 3, 3, 1)):
        super(ToRGB, self).__init__()
        if upsample:
            self.upsample = UpFirDnUpsample(resample_kernel, factor=2)
        else:
            self.upsample = None
        self.modulated_conv = ModulatedConv2d(in_channels,
                                              3,
                                              kernel_size=1,
                                              num_style_feat=num_style_feat,
                                              demodulate=False,
                                              sample_mode=None)
        self.bias = paddle.create_parameter(
            shape=[1, 3, 1, 1],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Constant(0))

    def forward(self, x, style, skip=None):
        """Forward function.

        Args:
            x (Tensor): Feature tensor with shape (b, c, h, w).
            style (Tensor): Tensor with shape (b, num_style_feat).
            skip (Tensor): Base/skip tensor. Default: None.

        Returns:
            Tensor: RGB images.
        """
        out = self.modulated_conv(x, style)
        out = out + self.bias
        if skip is not None:
            if self.upsample:
                skip = self.upsample(skip)
            out = out + skip
        return out


class ConstantInput(nn.Layer):
    """Constant input.

    Args:
        num_channel (int): Channel number of constant input.
        size (int): Spatial size of constant input.
    """
    def __init__(self, num_channel, size):
        super(ConstantInput, self).__init__()
        self.weight = paddle.create_parameter(
            shape=[1, num_channel, size, size],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Normal())

    def forward(self, batch):
        out = paddle.tile(self.weight, repeat_times=[batch, 1, 1, 1])
        return out


class FusedLeakyReLU(nn.Layer):
    def __init__(self, channel, bias=True, negative_slope=0.2, scale=2**0.5):
        super().__init__()
        if bias:
            self.bias = self.create_parameter(
                (channel, ), default_initializer=nn.initializer.Constant(0.0))
        else:
            self.bias = None
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope,
                                self.scale)


def fused_leaky_relu(input, bias=None, negative_slope=0.2, scale=2**0.5):
    if bias is not None:
        rest_dim = [1] * (len(input.shape) - len(bias.shape) - 1)
        return F.leaky_relu(input + bias.reshape([1, bias.shape[0], *rest_dim]),
                            negative_slope=0.2) * scale
    else:
        return F.leaky_relu(input, negative_slope=0.2) * scale


class ScaledLeakyReLU(nn.Layer):
    """Scaled LeakyReLU.

    Args:
        negative_slope (float): Negative slope. Default: 0.2.
    """
    def __init__(self, negative_slope=0.2):
        super(ScaledLeakyReLU, self).__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        out = F.leaky_relu(x, negative_slope=self.negative_slope)
        return out * math.sqrt(2)


class EqualConv2d(nn.Layer):
    """Equalized Linear as StyleGAN2.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        kernel_size (int): Size of the convolving kernel.
        stride (int): Stride of the convolution. Default: 1
        padding (int): Zero-padding added to both sides of the input.
            Default: 0.
        bias (bool): If ``True``, adds a learnable bias to the output.
            Default: ``True``.
        bias_init_val (float): Bias initialized value. Default: 0.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=True,
                 bias_init_val=0):
        super(EqualConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.scale = 1 / math.sqrt(in_channels * kernel_size**2)
        x = paddle.ones([out_channels, in_channels, kernel_size, kernel_size],
                        dtype="float32")
        self.weight = paddle.create_parameter(
            shape=[out_channels, in_channels, kernel_size, kernel_size],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Normal())
        if bias:
            self.bias = paddle.create_parameter(
                shape=[out_channels],
                dtype='float32',
                default_initializer=paddle.nn.initializer.Constant(
                    bias_init_val))
        else:
            pass
            self.bias = None

    def forward(self, x):
        out = F.conv2d(x,
                       self.weight * self.scale,
                       bias=self.bias,
                       stride=self.stride,
                       padding=self.padding)
        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}(in_channels={self.in_channels}, \
            out_channels={self.out_channels}, kernel_size={self.kernel_size}, \
            stride={self.stride}, padding={self.padding}, \
            bias={self.bias is not None})')


class ConvLayer(nn.Sequential):
    """Conv Layer used in StyleGAN2 Discriminator.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        kernel_size (int): Kernel size.
        downsample (bool): Whether downsample by a factor of 2.
            Default: False.
        resample_kernel (list[int]): A list indicating the 1D resample
            kernel magnitude. A cross production will be applied to
            extent 1D resample kenrel to 2D resample kernel.
            Default: (1, 3, 3, 1).
        bias (bool): Whether with bias. Default: True.
        activate (bool): Whether use activateion. Default: True.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 downsample=False,
                 resample_kernel=(1, 3, 3, 1),
                 bias=True,
                 activate=True):
        layers = []
        if downsample:
            layers.append(
                UpFirDnSmooth(resample_kernel,
                              upsample_factor=1,
                              downsample_factor=2,
                              kernel_size=kernel_size))
            stride = 2
            self.padding = 0
        else:
            stride = 1
            self.padding = kernel_size // 2
        layers.append(
            EqualConv2d(in_channels,
                        out_channels,
                        kernel_size,
                        stride=stride,
                        padding=self.padding,
                        bias=bias and not activate))
        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channels))
            else:
                layers.append(ScaledLeakyReLU(0.2))
        super(ConvLayer, self).__init__(*layers)


class ResBlock(nn.Layer):
    """Residual block used in StyleGAN2 Discriminator.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        resample_kernel (list[int]): A list indicating the 1D resample
            kernel magnitude. A cross production will be applied to
            extent 1D resample kenrel to 2D resample kernel.
            Default: (1, 3, 3, 1).
    """
    def __init__(self, in_channels, out_channels, resample_kernel=(1, 3, 3, 1)):
        super(ResBlock, self).__init__()
        self.conv1 = ConvLayer(in_channels,
                               in_channels,
                               3,
                               bias=True,
                               activate=True)
        self.conv2 = ConvLayer(in_channels,
                               out_channels,
                               3,
                               downsample=True,
                               resample_kernel=resample_kernel,
                               bias=True,
                               activate=True)
        self.skip = ConvLayer(in_channels,
                              out_channels,
                              1,
                              downsample=True,
                              resample_kernel=resample_kernel,
                              bias=False,
                              activate=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        skip = self.skip(x)
        out = (out + skip) / math.sqrt(2.)
        return out
