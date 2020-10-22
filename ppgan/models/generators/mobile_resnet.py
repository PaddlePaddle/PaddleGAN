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
import paddle
import paddle.nn as nn
import functools
from ...modules.norm import build_norm_layer
from .builder import GENERATORS


@GENERATORS.register()
class MobileResnetGenerator(nn.Layer):
    def __init__(self,
                 input_channel,
                 output_nc,
                 ngf=64,
                 norm_type='instance',
                 use_dropout=False,
                 n_blocks=9,
                 padding_type='reflect'):
        super(MobileResnetGenerator, self).__init__()

        norm_layer = build_norm_layer(norm_type)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2D
        else:
            use_bias = norm_layer == nn.InstanceNorm2D

        self.model = nn.LayerList([
            nn.ReflectionPad2d([3, 3, 3, 3]),
            nn.Conv2D(input_channel,
                      int(ngf),
                      kernel_size=7,
                      padding=0,
                      bias_attr=use_bias),
            norm_layer(ngf),
            nn.ReLU()
        ])

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            self.model.extend([
                nn.Conv2D(ngf * mult,
                          ngf * mult * 2,
                          kernel_size=3,
                          stride=2,
                          padding=1,
                          bias_attr=use_bias),
                norm_layer(ngf * mult * 2),
                nn.ReLU()
            ])

        mult = 2**n_downsampling

        for i in range(n_blocks):
            self.model.extend([
                MobileResnetBlock(ngf * mult,
                                  ngf * mult,
                                  padding_type=padding_type,
                                  norm_layer=norm_layer,
                                  use_dropout=use_dropout,
                                  use_bias=use_bias)
            ])

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            output_size = (i + 1) * 128
            self.model.extend([
                nn.Conv2DTranspose(ngf * mult,
                                   int(ngf * mult / 2),
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1,
                                   bias_attr=use_bias),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU()
            ])

        self.model.extend([nn.ReflectionPad2d([3, 3, 3, 3])])
        self.model.extend([nn.Conv2D(ngf, output_nc, kernel_size=7, padding=0)])
        self.model.extend([nn.Tanh()])

    def forward(self, inputs):
        y = inputs
        for sublayer in self.model:
            y = sublayer(y)
        return y


class MobileResnetBlock(nn.Layer):
    def __init__(self, in_c, out_c, padding_type, norm_layer, use_dropout,
                 use_bias):
        super(MobileResnetBlock, self).__init__()
        self.padding_type = padding_type
        self.use_dropout = use_dropout
        self.conv_block = nn.LayerList([])

        p = 0
        if self.padding_type == 'reflect':
            self.conv_block.extend([nn.Pad2D([1, 1, 1, 1], mode='reflect')])
        elif self.padding_type == 'replicate':
            self.conv_block.extend([nn.Pad2D([1, 1, 1, 1], mode='replicate')])
        elif self.padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' %
                                      self.padding_type)

        self.conv_block.extend([
            SeparableConv2D(num_channels=in_c,
                            num_filters=out_c,
                            filter_size=3,
                            padding=p,
                            stride=1),
            norm_layer(out_c),
            nn.ReLU()
        ])

        self.conv_block.extend([nn.Dropout(0.5)])

        if self.padding_type == 'reflect':
            self.conv_block.extend([nn.ReflectionPad2d([1, 1, 1, 1])])
        elif self.padding_type == 'replicate':
            self.conv_block.extend([nn.ReplicationPad2d([1, 1, 1, 1])])
        elif self.padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' %
                                      self.padding_type)

        self.conv_block.extend([
            SeparableConv2D(num_channels=out_c,
                            num_filters=in_c,
                            filter_size=3,
                            padding=p,
                            stride=1),
            norm_layer(in_c)
        ])

    def forward(self, inputs):
        y = inputs
        for sublayer in self.conv_block:
            y = sublayer(y)
        out = inputs + y
        return out


class SeparableConv2D(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 padding=0,
                 norm_layer=nn.InstanceNorm2D,
                 use_bias=True,
                 scale_factor=1,
                 stddev=0.02):
        super(SeparableConv2D, self).__init__()

        self.conv = nn.LayerList([
            nn.Conv2D(
                in_channels=num_channels,
                out_channels=num_channels * scale_factor,
                kernel_size=filter_size,
                stride=stride,
                padding=padding,
                groups=num_channels,
                weight_attr=paddle.ParamAttr(
                    initializer=nn.initializer.Normal(loc=0.0, scale=stddev)),
                bias_attr=use_bias)
        ])

        self.conv.extend([norm_layer(num_channels * scale_factor)])

        self.conv.extend([
            nn.Conv2D(
                in_channels=num_channels * scale_factor,
                out_channels=num_filters,
                kernel_size=1,
                stride=1,
                weight_attr=paddle.ParamAttr(
                    initializer=nn.initializer.Normal(loc=0.0, scale=stddev)),
                bias_attr=use_bias)
        ])

    def forward(self, inputs):
        for sublayer in self.conv:
            inputs = sublayer(inputs)
        return inputs
