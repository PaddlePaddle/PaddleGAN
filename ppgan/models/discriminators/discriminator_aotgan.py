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
from paddle.nn.utils import spectral_norm

from .builder import DISCRIMINATORS

@DISCRIMINATORS.register()
class Discriminator(nn.Layer):
    def __init__(self, inc = 3):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2D(inc, 64, 4, 2, 1, bias_attr=False)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2D(64, 128, 4, 2, 1, bias_attr=False)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2D(128, 256, 4, 2, 1, bias_attr=False)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2D(256, 512, 4, 1, 1, bias_attr=False)),
            nn.LeakyReLU(0.2),
            nn.Conv2D(512, 1, 4, 1, 1)
        )

    def forward(self, x):
        feat = self.conv(x)
        return feat
