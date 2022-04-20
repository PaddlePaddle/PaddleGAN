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

# code was heavily based on https://github.com/csdwren/PReNet
# Users should be careful about adopting these functions in any commercial matters.

import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .builder import GENERATORS


def convWithBias(in_channels, out_channels, kernel_size, stride, padding):
    """ Obtain a 2d convolution layer with bias and initialized by KaimingUniform
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Convolution kernel size
        stride (int): Convolution stride
        padding (int|tuple): Convolution padding.
    """
    if isinstance(kernel_size, int):
        fan_in = kernel_size * kernel_size * in_channels
    else:
        fan_in = kernel_size[0] * kernel_size[1] * in_channels
    bound = 1 / math.sqrt(fan_in)
    bias_attr = paddle.framework.ParamAttr(
        initializer=nn.initializer.Uniform(-bound, bound))
    weight_attr = paddle.framework.ParamAttr(
        initializer=nn.initializer.KaimingUniform(fan_in=6 * fan_in))
    conv = nn.Conv2D(in_channels,
                     out_channels,
                     kernel_size,
                     stride,
                     padding,
                     weight_attr=weight_attr,
                     bias_attr=bias_attr)
    return conv


@GENERATORS.register()
class PReNet(nn.Layer):
    """
    Args:
        recurrent_iter (int): Number of iterations.
            Default: 6.
        use_GPU (bool): whether use gpu or not .
            Default: True.
    """

    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(PReNet, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv0 = nn.Sequential(convWithBias(6, 32, 3, 1, 1), nn.ReLU())
        self.res_conv1 = nn.Sequential(convWithBias(32, 32, 3, 1, 1), nn.ReLU(),
                                       convWithBias(32, 32, 3, 1, 1), nn.ReLU())
        self.res_conv2 = nn.Sequential(convWithBias(32, 32, 3, 1, 1), nn.ReLU(),
                                       convWithBias(32, 32, 3, 1, 1), nn.ReLU())
        self.res_conv3 = nn.Sequential(convWithBias(32, 32, 3, 1, 1), nn.ReLU(),
                                       convWithBias(32, 32, 3, 1, 1), nn.ReLU())
        self.res_conv4 = nn.Sequential(convWithBias(32, 32, 3, 1, 1), nn.ReLU(),
                                       convWithBias(32, 32, 3, 1, 1), nn.ReLU())
        self.res_conv5 = nn.Sequential(convWithBias(32, 32, 3, 1, 1), nn.ReLU(),
                                       convWithBias(32, 32, 3, 1, 1), nn.ReLU())
        self.conv_i = nn.Sequential(convWithBias(32 + 32, 32, 3, 1, 1),
                                    nn.Sigmoid())
        self.conv_f = nn.Sequential(convWithBias(32 + 32, 32, 3, 1, 1),
                                    nn.Sigmoid())
        self.conv_g = nn.Sequential(convWithBias(32 + 32, 32, 3, 1, 1),
                                    nn.Tanh())
        self.conv_o = nn.Sequential(convWithBias(32 + 32, 32, 3, 1, 1),
                                    nn.Sigmoid())
        self.conv = nn.Sequential(convWithBias(32, 3, 3, 1, 1), )

    def forward(self, input):
        batch_size, row, col = input.shape[0], input.shape[2], input.shape[3]

        x = input

        h = paddle.to_tensor(paddle.zeros(shape=(batch_size, 32, row, col),
                                          dtype='float32'),
                             stop_gradient=False)
        c = paddle.to_tensor(paddle.zeros(shape=(batch_size, 32, row, col),
                                          dtype='float32'),
                             stop_gradient=False)

        x_list = []
        for _ in range(self.iteration):
            x = paddle.concat((input, x), 1)
            x = self.conv0(x)

            x = paddle.concat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * paddle.tanh(c)

            x = h
            resx = x
            x = F.relu(self.res_conv1(x) + resx)
            resx = x
            x = F.relu(self.res_conv2(x) + resx)
            resx = x
            x = F.relu(self.res_conv3(x) + resx)
            resx = x
            x = F.relu(self.res_conv4(x) + resx)
            resx = x
            x = F.relu(self.res_conv5(x) + resx)
            x = self.conv(x)

            x = x + input
            x_list.append(x)
        return x
