# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

# code was heavily based on https://github.com/aidotse/Team-Haste
# MIT License
# Copyright (c) 2020 AI Sweden

import paddle
import functools
import numpy as np
import paddle.nn as nn
import paddle.nn.functional as F

from paddle.nn import BatchNorm2D
from ...modules.norm import build_norm_layer

from .builder import DISCRIMINATORS


@DISCRIMINATORS.register()
class DCDiscriminator(nn.Layer):
    """Defines a DCGAN discriminator"""
    def __init__(self, input_nc, ndf=64, norm_type='instance'):
        """Construct a DCGAN discriminator

        Parameters:
            input_nc (int): the number of channels in input images
            ndf (int): the number of filters in the last conv layer
            norm_type (str): normalization layer type
        """
        super(DCDiscriminator, self).__init__()
        norm_layer = build_norm_layer(norm_type)
        if type(norm_layer) == functools.partial:
            # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.BatchNorm2D
        else:
            use_bias = norm_layer == nn.BatchNorm2D

        kw = 4
        padw = 1

        sequence = [
            nn.Conv2D(input_nc,
                      ndf,
                      kernel_size=kw,
                      stride=2,
                      padding=padw,
                      bias_attr=use_bias),
            nn.LeakyReLU(0.2)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        n_downsampling = 4

        # gradually increase the number of filters
        for n in range(1, n_downsampling):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            if norm_type == 'batch':
                sequence += [
                    nn.Conv2D(ndf * nf_mult_prev,
                              ndf * nf_mult,
                              kernel_size=kw,
                              stride=2,
                              padding=padw),
                    BatchNorm2D(ndf * nf_mult),
                    nn.LeakyReLU(0.2)
                ]
            else:
                sequence += [
                    nn.Conv2D(ndf * nf_mult_prev,
                              ndf * nf_mult,
                              kernel_size=kw,
                              stride=2,
                              padding=padw,
                              bias_attr=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2)
                ]

        nf_mult_prev = nf_mult

        # output 1 channel prediction map
        sequence += [
            nn.Conv2D(ndf * nf_mult_prev,
                      1,
                      kernel_size=kw,
                      stride=1,
                      padding=0)
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)
