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

# code was based on https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

import paddle
import paddle.nn as nn
import functools

from ...modules.norm import build_norm_layer

from .builder import GENERATORS


@GENERATORS.register()
class ResnetGenerator(nn.Layer):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)

    Args:
            input_nc (int): the number of channels in input images
            output_nc (int): the number of channels in output images
            ngf (int): the number of filters in the last conv layer
            norm_type (str): the name of the normalization layer: batch | instance | none
            use_dropout (bool): if use dropout layers
            n_blocks (int): the number of ResNet blocks
            padding_type (str): the name of padding layer in conv layers: reflect | replicate | zero

    """
    def __init__(self,
                 input_nc,
                 output_nc,
                 ngf=64,
                 norm_type='instance',
                 use_dropout=False,
                 n_blocks=6,
                 padding_type='reflect'):

        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()

        norm_layer = build_norm_layer(norm_type)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2D
        else:
            use_bias = norm_layer == nn.InstanceNorm2D

        model = [
            nn.Pad2D(padding=[3, 3, 3, 3], mode="reflect"),
            nn.Conv2D(input_nc,
                      ngf,
                      kernel_size=7,
                      padding=0,
                      bias_attr=use_bias),
            norm_layer(ngf),
            nn.ReLU()
        ]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2**i
            model += [
                nn.Conv2D(ngf * mult,
                          ngf * mult * 2,
                          kernel_size=3,
                          stride=2,
                          padding=1,
                          bias_attr=use_bias),
                norm_layer(ngf * mult * 2),
                nn.ReLU()
            ]

        mult = 2**n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [
                ResnetBlock(ngf * mult,
                            padding_type=padding_type,
                            norm_layer=norm_layer,
                            use_dropout=use_dropout,
                            use_bias=use_bias)
            ]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2**(n_downsampling - i)
            model += [
                nn.Conv2DTranspose(ngf * mult,
                                   int(ngf * mult / 2),
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1,
                                   bias_attr=use_bias),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU()
            ]
        model += [nn.Pad2D(padding=[3, 3, 3, 3], mode="reflect")]
        model += [nn.Conv2D(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        """Standard forward"""
        return self.model(x)


class ResnetBlock(nn.Layer):
    """Define a Resnet block"""
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer,
                                                use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout,
                         use_bias):
        """Construct a convolutional block.

        Args:
            dim (int): the number of channels in the conv layer.
            padding_type (str): the name of padding layer: reflect | replicate | zero.
            norm_layer (paddle.nn.Layer): normalization layer.
            use_dropout (bool): whether to  use dropout layers.
            use_bias (bool): whether to use the conv layer bias or not.

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type in ['reflect', 'replicate']:
            conv_block += [nn.Pad2D(padding=[1, 1, 1, 1], mode=padding_type)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' %
                                      padding_type)

        conv_block += [
            nn.Conv2D(dim, dim, kernel_size=3, padding=p, bias_attr=use_bias),
            norm_layer(dim),
            nn.ReLU()
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type in ['reflect', 'replicate']:
            conv_block += [nn.Pad2D(padding=[1, 1, 1, 1], mode=padding_type)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' %
                                      padding_type)
        conv_block += [
            nn.Conv2D(dim, dim, kernel_size=3, padding=p, bias_attr=use_bias),
            norm_layer(dim)
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out
