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

# code was heavily based on https://github.com/aidotse/Team-Haste
# MIT License
# Copyright (c) 2020 AI Sweden

import paddle
import paddle.nn as nn
import functools

from paddle.nn import BatchNorm2D
from ...modules.norm import build_norm_layer

from .builder import GENERATORS


@GENERATORS.register()
class DCGenerator(nn.Layer):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    """
    def __init__(self,
                 input_nz,
                 input_nc,
                 output_nc,
                 ngf=64,
                 norm_type='batch',
                 padding_type='reflect'):
        """Construct a DCGenerator generator

        Args:
            input_nz (int): the number of dimension in input noise
            input_nc (int): the number of channels in input images
            output_nc (int): the number of channels in output images
            ngf (int): the number of filters in the last conv layer
            norm_layer: normalization layer
            padding_type (str): the name of padding layer in conv layers: reflect | replicate | zero
        """
        super(DCGenerator, self).__init__()

        norm_layer = build_norm_layer(norm_type)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.BatchNorm2D
        else:
            use_bias = norm_layer == nn.BatchNorm2D

        mult = 8
        n_downsampling = 4

        if norm_type == 'batch':
            model = [
                nn.Conv2DTranspose(input_nz,
                                   ngf * mult,
                                   kernel_size=4,
                                   stride=1,
                                   padding=0,
                                   bias_attr=use_bias),
                BatchNorm2D(ngf * mult),
                nn.ReLU()
            ]
        else:
            model = [
                nn.Conv2DTranspose(input_nz,
                                   ngf * mult,
                                   kernel_size=4,
                                   stride=1,
                                   padding=0,
                                   bias_attr=use_bias),
                norm_layer(ngf * mult),
                nn.ReLU()
            ]

        # add upsampling layers
        for i in range(1, n_downsampling):
            mult = 2**(n_downsampling - i)
            output_size = 2**(i + 2)
            if norm_type == 'batch':
                model += [
                    nn.Conv2DTranspose(ngf * mult,
                                       ngf * mult // 2,
                                       kernel_size=4,
                                       stride=2,
                                       padding=1,
                                       bias_attr=use_bias),
                    BatchNorm2D(ngf * mult // 2),
                    nn.ReLU()
                ]
            else:
                model += [
                    nn.Conv2DTranspose(ngf * mult,
                                       int(ngf * mult // 2),
                                       kernel_size=4,
                                       stride=2,
                                       padding=1,
                                       bias_attr=use_bias),
                    norm_layer(int(ngf * mult // 2)),
                    nn.ReLU()
                ]

        output_size = 2**(6)
        model += [
            nn.Conv2DTranspose(ngf,
                               output_nc,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias_attr=use_bias),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        """Standard forward"""
        return self.model(x)
