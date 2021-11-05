#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

# code was heavily based on https://github.com/rosinality/stylegan2-pytorch
# MIT License
# Copyright (c) 2019 Kim Seonghyeon

import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .builder import DISCRIMINATORS
from ...modules.equalized import EqualLinear, EqualConv2D
from ...modules.fused_act import FusedLeakyReLU
from ...modules.upfirdn2d import Upfirdn2dBlur


class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Upfirdn2dBlur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2D(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            ))

        if activate:
            layers.append(FusedLeakyReLU(out_channel, bias=bias))

        super().__init__(*layers)


class ResBlock(nn.Layer):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

        self.skip = ConvLayer(in_channel,
                              out_channel,
                              1,
                              downsample=True,
                              activate=False,
                              bias=False)

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


# temporally solve pow double grad problem
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
class StyleGANv2Discriminator(nn.Layer):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [ConvLayer(3, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2**(i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4,
                        channels[4],
                        activation="fused_lrelu"),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        out = self.convs(input)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.reshape((group, -1, self.stddev_feat,
                              channel // self.stddev_feat, height, width))
        stddev = paddle.sqrt(var(stddev, 0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdim=True).squeeze(2)
        stddev = stddev.tile((group, 1, height, width))
        out = paddle.concat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.reshape((batch, -1))
        out = self.final_linear(out)

        return out
