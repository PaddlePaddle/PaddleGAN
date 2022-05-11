# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

import itertools
import paddle.nn as nn
import math
from .builder import GENERATORS
from ...modules.equalized import EqualLinear_gpen as EqualLinear
from .generator_gpen import StyleGANv2Generator
from ..discriminators.discriminator_styleganv2 import ConvLayer

@GENERATORS.register()
class GPEN(nn.Layer):
    def __init__(
            self,
            size,
            style_dim,
            n_mlp,
            channel_multiplier=2,
            narrow=1,
            blur_kernel=[1, 3, 3, 1],
            lr_mlp=0.01,
            is_concat=True,
    ):
        super(GPEN, self).__init__()
        channels = {
            4: int(512 * narrow),
            8: int(512 * narrow),
            16: int(512 * narrow),
            32: int(512 * narrow),
            64: int(256 * channel_multiplier * narrow),
            128: int(128 * channel_multiplier * narrow),
            256: int(64 * channel_multiplier * narrow),
            512: int(32 * channel_multiplier * narrow),
            1024: int(16 * channel_multiplier * narrow),
            2048: int(8 * channel_multiplier * narrow)
        }
        self.log_size = int(math.log(size, 2))
        self.generator = StyleGANv2Generator(size,
                                             style_dim,
                                             n_mlp,
                                             channel_multiplier=channel_multiplier,
                                             narrow=narrow,
                                             blur_kernel=blur_kernel,
                                             lr_mlp=lr_mlp,
                                             is_concat=is_concat)

        conv = [ConvLayer(3, channels[size], 1)]
        self.ecd0 = nn.Sequential(*conv)
        in_channel = channels[size]

        self.names = ['ecd%d' % i for i in range(self.log_size - 1)]
        for i in range(self.log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            conv = [ConvLayer(in_channel, out_channel, 3, downsample=True)]
            setattr(self, self.names[self.log_size - i + 1], nn.Sequential(*conv))
            in_channel = out_channel
        self.final_linear = nn.Sequential(EqualLinear(channels[4] * 4 * 4, style_dim, activation='fused_lrelu'))

    def forward(self,
                inputs,
                return_latents=False,
                inject_index=None,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                ):
        noise = []
        for i in range(self.log_size - 1):
            ecd = getattr(self, self.names[i])
            inputs = ecd(inputs)
            noise.append(inputs)
        inputs = inputs.reshape([inputs.shape[0], -1])
        outs = self.final_linear(inputs)
        noise = list(itertools.chain.from_iterable(itertools.repeat(x, 2) for x in noise))[::-1]
        outs = self.generator([outs], return_latents, inject_index, truncation,
                              truncation_latent, input_is_latent,
                              noise=noise[1:])
        return outs


