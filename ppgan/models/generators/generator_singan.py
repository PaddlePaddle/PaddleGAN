#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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
# code was based on https://github.com/tamarott/SinGAN

import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from .builder import GENERATORS


class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvBlock,self).__init__()
        self.add_sublayer('conv', nn.Conv2D(in_channel ,out_channel, ker_size, stride, padd)),
        self.add_sublayer('norm', nn.BatchNorm2D(out_channel)),
        self.add_sublayer('LeakyRelu', nn.LeakyReLU(0.2))

class GeneratorConcatSkip2CleanAdd(nn.Layer):
    def __init__(self, nfc=32, min_nfc=32, input_nc=3, num_layers=5, ker_size=3, padd_size=0):
        super(GeneratorConcatSkip2CleanAdd, self).__init__()
        self.head = ConvBlock(input_nc, nfc, ker_size, padd_size, 1)
        self.body = nn.Sequential()
        for i in range(num_layers - 2):
            N = int(nfc / pow(2, i + 1))
            block = ConvBlock(max(2 * N, min_nfc), max(N, min_nfc), ker_size, padd_size, 1)
            self.body.add_sublayer('block%d' % (i + 1), block)
        self.tail = nn.Sequential(
            nn.Conv2D(max(N, min_nfc), input_nc, ker_size, 1, padd_size),
            nn.Tanh())
    def forward(self, x, y):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        ind = int((y.shape[2] - x.shape[2]) / 2)
        y = y[:, :, ind: (y.shape[2] - ind), ind: (y.shape[3] - ind)]
        return x + y

@GENERATORS.register()
class SinGANGenerator(nn.Layer):
    def __init__(self, 
                 scale_num, 
                 coarsest_shape, 
                 nfc_init=32, 
                 min_nfc_init=32, 
                 input_nc=3, 
                 num_layers=5, 
                 ker_size=3, 
                 noise_zero_pad=True):
        super().__init__()
        nfc_list = [min(nfc_init * pow(2, math.floor(i / 4)), 128) for i in range(scale_num)]
        min_nfc_list = [min(min_nfc_init * pow(2, math.floor(i / 4)), 128) for i in range(scale_num)]
        self.generators = nn.LayerList([
            GeneratorConcatSkip2CleanAdd(
                nfc, min_nfc, input_nc, num_layers, 
                ker_size, 0
            ) for nfc, min_nfc in zip(nfc_list, min_nfc_list)])
        self._scale_num = scale_num
        self._pad_size = int((ker_size - 1) / 2 * num_layers)
        self.noise_pad = nn.Pad2D(self._pad_size if noise_zero_pad else 0)
        self.image_pad = nn.Pad2D(self._pad_size)
        self._noise_zero_pad = noise_zero_pad
        self._coarsest_shape = coarsest_shape
        self.register_buffer('scale_num', paddle.to_tensor(scale_num, 'int32'), True)
        self.register_buffer('coarsest_shape', paddle.to_tensor(coarsest_shape, 'int32'), True)
        self.register_buffer('nfc_init', paddle.to_tensor(nfc_init, 'int32'), True)
        self.register_buffer('min_nfc_init', paddle.to_tensor(min_nfc_init, 'int32'), True)
        self.register_buffer('num_layers', paddle.to_tensor(num_layers, 'int32'), True)
        self.register_buffer('ker_size', paddle.to_tensor(ker_size, 'int32'), True)
        self.register_buffer('noise_zero_pad', paddle.to_tensor(noise_zero_pad, 'bool'), True)
        self.register_buffer('sigma', paddle.ones([scale_num]), True)
        self.register_buffer('scale_factor', paddle.ones([1]), True)
        self.register_buffer(
            'z_fixed', 
            paddle.randn(
                F.pad(
                    paddle.zeros(coarsest_shape), 
                    [0 if noise_zero_pad else self._pad_size] * 4).shape), True)

    def forward(self, z_pyramid, x_prev, stop_scale, start_scale=0):
        stop_scale %= self._scale_num
        start_scale %= self._scale_num
        for i, scale in enumerate(range(start_scale, stop_scale + 1)):
            x_prev = self.image_pad(x_prev)
            z = self.noise_pad(z_pyramid[i] * self.sigma[scale]) + x_prev
            x_prev = self.generators[scale](
                z.detach(), 
                x_prev.detach()
            )
            if scale < stop_scale:
                x_prev = F.interpolate(x_prev, 
                    F.pad(z_pyramid[i + 1], [0 if self._noise_zero_pad else -self._pad_size] * 4).shape[-2:],
                    None, 'bicubic')
        return x_prev
