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

from .fused_act import fused_leaky_relu


class EqualConv2D(nn.Layer):
    """This convolutional layer class stabilizes the learning rate changes of its parameters.
    Equalizing learning rate keeps the weights in the network at a similar scale during training.
    """
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=True):
        super().__init__()

        self.weight = self.create_parameter(
            (out_channel, in_channel, kernel_size, kernel_size),
            default_initializer=nn.initializer.Normal())
        self.scale = 1 / math.sqrt(in_channel * (kernel_size * kernel_size))

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = self.create_parameter((out_channel, ),
                                              nn.initializer.Constant(0.0))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


class EqualLinear(nn.Layer):
    """This linear layer class stabilizes the learning rate changes of its parameters.
    Equalizing learning rate keeps the weights in the network at a similar scale during training.
    """
    def __init__(self,
                 in_dim,
                 out_dim,
                 bias=True,
                 bias_init=0,
                 lr_mul=1,
                 activation=None):
        super().__init__()

        self.weight = self.create_parameter(
            (in_dim, out_dim), default_initializer=nn.initializer.Normal())
        self.weight.set_value((self.weight / lr_mul))

        if bias:
            self.bias = self.create_parameter(
                (out_dim, ), nn.initializer.Constant(bias_init))

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
