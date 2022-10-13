#  Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.utils import spectral_norm

from .builder import GENERATORS

#  Aggregated Contextual Transformations Block
class AOTBlock(nn.Layer):
    def __init__(self, dim, rates):
        super(AOTBlock, self).__init__()

        self.rates = rates
        for i, rate in enumerate(rates):
            self.__setattr__(
                'block{}'.format(str(i).zfill(2)),
                nn.Sequential(
                    nn.Pad2D(rate, mode='reflect'),
                    nn.Conv2D(dim, dim//4, 3, 1, 0, dilation=int(rate)),
                    nn.ReLU()))
        self.fuse = nn.Sequential(
            nn.Pad2D(1, mode='reflect'),
            nn.Conv2D(dim, dim, 3, 1, 0, dilation=1))
        self.gate = nn.Sequential(
            nn.Pad2D(1, mode='reflect'),
            nn.Conv2D(dim, dim, 3, 1, 0, dilation=1))

    def forward(self, x):
        out = [self.__getattr__(f'block{str(i).zfill(2)}')(x) for i in range(len(self.rates))]
        out = paddle.concat(out, 1)
        out = self.fuse(out)
        mask = my_layer_norm(self.gate(x))
        mask = F.sigmoid(mask)
        return x * (1 - mask) + out * mask

def my_layer_norm(feat):
    mean = feat.mean((2, 3), keepdim=True)
    std = feat.std((2, 3), keepdim=True) + 1e-9
    feat = 2 * (feat - mean) / std - 1
    feat = 5 * feat
    return feat

class UpConv(nn.Layer):
    def __init__(self, inc, outc, scale=2):
        super(UpConv, self).__init__()
        self.scale = scale
        self.conv = nn.Conv2D(inc, outc, 3, 1, 1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True))

# generator
@GENERATORS.register()
class InpaintGenerator(nn.Layer):
    def __init__(self, rates, block_num):
        super(InpaintGenerator, self).__init__()

        self.encoder = nn.Sequential(
            nn.Pad2D(3, mode='reflect'),
            nn.Conv2D(4, 64, 7, 1, 0),
            nn.ReLU(),
            nn.Conv2D(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2D(128, 256, 4, 2, 1),
            nn.ReLU()
        )

        self.middle = nn.Sequential(*[AOTBlock(256, rates) for _ in range(block_num)])

        self.decoder = nn.Sequential(
            UpConv(256, 128),
            nn.ReLU(),
            UpConv(128, 64),
            nn.ReLU(),
            nn.Conv2D(64, 3, 3, 1, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = paddle.tanh(x)

        return x
