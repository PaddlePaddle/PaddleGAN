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
from paddle import nn
from paddle.nn import functional as F


class NormStyleCode(nn.Layer):
    def forward(self, x):
        """Normalize the style codes.

        Args:
            x (Tensor): Style codes with shape (b, c).

        Returns:
            Tensor: Normalized tensor.
        """
        return x * paddle.rsqrt(paddle.mean(x ** 2, axis=1, keepdim=\
            True) + 1e-08)


class ModulatedConv2d(nn.Layer):
    """Modulated Conv2d used in StyleGAN2.

    There is no bias in ModulatedConv2d.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        kernel_size (int): Size of the convolving kernel.
        num_style_feat (int): Channel number of style features.
        demodulate (bool): Whether to demodulate in the conv layer. Default: True.
        sample_mode (str | None): Indicating 'upsample', 'downsample' or None. Default: None.
        eps (float): A value added to the denominator for numerical stability. Default: 1e-8.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 num_style_feat,
                 demodulate=True,
                 sample_mode=None,
                 eps=1e-08):
        super(ModulatedConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.sample_mode = sample_mode
        self.eps = eps
        self.modulation = nn.Linear(num_style_feat, in_channels, bias_attr=True)
        # default_init_weights(self.modulation, scale=1, bias_fill=1, a=0,
        #     mode='fan_in', nonlinearity='linear')
        x=paddle.randn(shape=[1, out_channels, in_channels, kernel_size, kernel_size],dtype='float32')/math. \
            sqrt(in_channels * kernel_size ** 2)

        self.weight = paddle.create_parameter(
            shape=x.shape,
            dtype='float32',
            default_initializer=paddle.nn.initializer.Assign(x))
        self.weight.stop_gradient = False
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
        weight = self.weight * style
        if self.demodulate:
            demod = paddle.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps)
            weight = weight * demod.reshape([b, self.out_channels, 1, 1, 1])
        weight = weight.reshape(
            [b * self.out_channels, c, self.kernel_size, self.kernel_size])
        if self.sample_mode == 'upsample':
            x = F.interpolate(x,
                              scale_factor=2,
                              mode='bilinear',
                              align_corners=False)
        elif self.sample_mode == 'downsample':
            x = F.interpolate(x,
                              scale_factor=0.5,
                              mode='bilinear',
                              align_corners=False)
        b, c, h, w = x.shape
        x = x.reshape([1, b * c, h, w])
        out = paddle.nn.functional.conv2d(x,
                                          weight,
                                          padding=self.padding,
                                          groups=b)
        out = out.reshape([b, self.out_channels, *out.shape[2:4]])
        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}(in_channels={self.in_channels}, \
            out_channels={self.out_channels}, \
            kernel_size={self.kernel_size}, \
            demodulate={self.demodulate}, \
            sample_mode={self.sample_mode})')


class StyleConv(nn.Layer):
    """Style conv used in StyleGAN2.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        kernel_size (int): Size of the convolving kernel.
        num_style_feat (int): Channel number of style features.
        demodulate (bool): Whether demodulate in the conv layer. Default: True.
        sample_mode (str | None): Indicating 'upsample', 'downsample' or None. Default: None.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 num_style_feat,
                 demodulate=True,
                 sample_mode=None):
        super(StyleConv, self).__init__()
        self.modulated_conv = ModulatedConv2d(in_channels,
                                              out_channels,
                                              kernel_size,
                                              num_style_feat,
                                              demodulate=demodulate,
                                              sample_mode=sample_mode)

        x = paddle.zeros([1], dtype="float32")
        self.weight = paddle.create_parameter(
            x.shape,
            dtype='float32',
            default_initializer=paddle.nn.initializer.Assign(
                x))  # for noise injection
        x = paddle.zeros([1, out_channels, 1, 1], dtype="float32")
        self.bias = paddle.create_parameter(
            x.shape,
            dtype='float32',
            default_initializer=paddle.nn.initializer.Assign(x))
        self.activate = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, style, noise=None):
        out = self.modulated_conv(x, style) * 2**0.5
        if noise is None:
            b, _, h, w = out.shape
            noise = paddle.normal(shape=[b, 1, h, w])
        out = out + self.weight * noise
        out = out + self.bias
        out = self.activate(out)
        return out


class ToRGB(nn.Layer):
    """To RGB (image space) from features.

    Args:
        in_channels (int): Channel number of input.
        num_style_feat (int): Channel number of style features.
        upsample (bool): Whether to upsample. Default: True.
    """
    def __init__(self, in_channels, num_style_feat, upsample=True):
        super(ToRGB, self).__init__()
        self.upsample = upsample
        self.modulated_conv = ModulatedConv2d(in_channels,
                                              3,
                                              kernel_size=1,
                                              num_style_feat=num_style_feat,
                                              demodulate=False,
                                              sample_mode=None)
        x = paddle.zeros(shape=[1, 3, 1, 1], dtype='float32')
        self.bias = paddle.create_parameter(
            shape=x.shape,
            dtype='float32',
            default_initializer=paddle.nn.initializer.Assign(x))
        self.bias.stop_gradient = False

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
                skip = F.interpolate(skip,
                                     scale_factor=2,
                                     mode='bilinear',
                                     align_corners=False)
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
        x = paddle.randn(shape=[1, num_channel, size, size], dtype='float32')
        self.weight = paddle.create_parameter(
            shape=x.shape,
            dtype='float32',
            default_initializer=paddle.nn.initializer.Assign(x))
        self.weight.stop_gradient = False

    def forward(self, batch):
        out = paddle.tile(self.weight, repeat_times=[batch, 1, 1, 1])
        return out


class StyleGAN2GeneratorClean(nn.Layer):
    """Clean version of StyleGAN2 Generator.

    Args:
        out_size (int): The spatial size of outputs.
        num_style_feat (int): Channel number of style features. Default: 512.
        num_mlp (int): Layer number of MLP style layers. Default: 8.
        channel_multiplier (int): Channel multiplier for large networks of StyleGAN2. Default: 2.
        narrow (float): Narrow ratio for channels. Default: 1.0.
    """
    def __init__(self,
                 out_size,
                 num_style_feat=512,
                 num_mlp=8,
                 channel_multiplier=2,
                 narrow=1):
        super(StyleGAN2GeneratorClean, self).__init__()
        self.num_style_feat = num_style_feat
        style_mlp_layers = [NormStyleCode()]
        for i in range(num_mlp):
            style_mlp_layers.extend([
                nn.Linear(num_style_feat, num_style_feat, bias_attr=True),
                nn.LeakyReLU(negative_slope=0.2)
            ])
        self.style_mlp = nn.Sequential(*style_mlp_layers)
        # default_init_weights(self.style_mlp, scale=1, bias_fill=0, a=0.2,
        #     mode='fan_in', nonlinearity='leaky_relu')
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
                                     sample_mode=None)
        self.to_rgb1 = ToRGB(channels['4'], num_style_feat, upsample=False)
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
            self.noises.register_buffer(f'noise{layer_idx}',
                                        paddle.randn(shape=shape))
        for i in range(3, self.log_size + 1):
            out_channels = channels[f'{2 ** i}']
            self.style_convs.append(StyleConv(in_channels, out_channels,
                kernel_size=3, num_style_feat=num_style_feat, demodulate=\
                True, sample_mode='upsample'))
            self.style_convs.append(StyleConv(out_channels, out_channels,
                kernel_size=3, num_style_feat=num_style_feat, demodulate=\
                True, sample_mode=None))
            self.to_rgbs.append(
                ToRGB(out_channels, num_style_feat, upsample=True))
            in_channels = out_channels

    def make_noise(self):
        """Make noise for noise injection."""
        device = self.constant_input.weight.device
        noises = [paddle.randn(shape=[1, 1, 4, 4])]
        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(paddle.randn(shape=[1, 1, 2**i, 2**i]))
        return noises

    def get_latent(self, x):
        return self.style_mlp(x)

    def mean_latent(self, num_latent):
        latent_in = paddle.randn(shape=[num_latent, self.num_style_feat])
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
        """Forward function for StyleGAN2GeneratorClean.

        Args:
            styles (list[Tensor]): Sample codes of styles.
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
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:
                latent = styles[0]
        elif len(styles) == 2:
            if inject_index is None:
                inject_index = random.randint(1, self.num_latent - 1)
            latent1 = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(
                1, self.num_latent - inject_index, 1)
            latent = paddle.concat([latent1, latent2], axis=1)
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
