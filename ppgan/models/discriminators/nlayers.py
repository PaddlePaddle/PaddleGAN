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

# code was heavily based on https://github.com/wtjiang98/PSGAN
# MIT License 
# Copyright (c) 2020 Wentao Jiang

import paddle
import functools
import numpy as np
import paddle.nn as nn
import paddle.nn.functional as F

from ...modules.nn import Spectralnorm
from ...modules.norm import build_norm_layer

from .builder import DISCRIMINATORS


@DISCRIMINATORS.register()
class NLayerDiscriminator(nn.Layer):
    """Defines a PatchGAN discriminator"""
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_type='instance', use_sigmoid=False):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_type (str)      -- normalization layer type
            use_sigmoid (bool)   -- whether use sigmoid at last
        """
        super(NLayerDiscriminator, self).__init__()
        norm_layer = build_norm_layer(norm_type)
        if type(
                norm_layer
        ) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2D
        else:
            use_bias = norm_layer == nn.InstanceNorm2D

        kw = 4
        padw = 1

        if norm_type == 'spectral':
            sequence = [
                Spectralnorm(
                    nn.Conv2D(input_nc,
                              ndf,
                              kernel_size=kw,
                              stride=2,
                              padding=padw)),
                nn.LeakyReLU(0.01)
            ]
        else:
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
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            if norm_type == 'spectral':
                sequence += [
                    Spectralnorm(
                        nn.Conv2D(ndf * nf_mult_prev,
                                  ndf * nf_mult,
                                  kernel_size=kw,
                                  stride=2,
                                  padding=padw)),
                    nn.LeakyReLU(0.01)
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
        nf_mult = min(2**n_layers, 8)
        if norm_type == 'spectral':
            sequence += [
                Spectralnorm(
                    nn.Conv2D(ndf * nf_mult_prev,
                              ndf * nf_mult,
                              kernel_size=kw,
                              stride=1,
                              padding=padw)),
                nn.LeakyReLU(0.01)
            ]
        else:
            sequence += [
                nn.Conv2D(ndf * nf_mult_prev,
                          ndf * nf_mult,
                          kernel_size=kw,
                          stride=1,
                          padding=padw,
                          bias_attr=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2)
            ]

        if norm_type == 'spectral':
            sequence += [
                Spectralnorm(
                    nn.Conv2D(ndf * nf_mult,
                              1,
                              kernel_size=kw,
                              stride=1,
                              padding=padw,
                              bias_attr=False))
            ]  # output 1 channel prediction map
        else:
            sequence += [
                nn.Conv2D(ndf * nf_mult,
                          1,
                          kernel_size=kw,
                          stride=1,
                          padding=padw)
            ]  # output 1 channel prediction map

        self.model = nn.Sequential(*sequence)
        self.final_act = F.sigmoid if use_sigmoid else (lambda x:x)

    def forward(self, input):
        """Standard forward."""
        return self.final_act(self.model(input))


@DISCRIMINATORS.register()
class NLayerDiscriminatorWithClassification(NLayerDiscriminator):
    def __init__(self, input_nc, n_class=10, **kwargs):
        input_nc = input_nc + n_class
        super(NLayerDiscriminatorWithClassification, self).__init__(input_nc, **kwargs)

        self.n_class = n_class
    
    def forward(self, x, class_id):
        if self.n_class > 0:
            class_id = (class_id % self.n_class).detach()
            class_id = F.one_hot(class_id, self.n_class).astype('float32')
            class_id = class_id.reshape([x.shape[0], -1, 1, 1])
            class_id = class_id.tile([1,1,*x.shape[2:]])
            x = paddle.concat([x, class_id], 1)
        
        return super(NLayerDiscriminatorWithClassification, self).forward(x)

