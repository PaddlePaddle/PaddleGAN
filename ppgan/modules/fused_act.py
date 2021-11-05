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

# code was heavily based on https://github.com/rosinality/stylegan2-pytorch
# MIT License
# Copyright (c) 2019 Kim Seonghyeon

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


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
        return (F.leaky_relu(input + bias.reshape(
            (1, bias.shape[0], *rest_dim)),
                             negative_slope=0.2) * scale)

    else:
        return F.leaky_relu(input, negative_slope=0.2) * scale
