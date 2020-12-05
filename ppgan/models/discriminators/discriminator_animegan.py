#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import paddle.nn as nn
import paddle.nn.functional as F

from .builder import DISCRIMINATORS
from ...modules.utils import spectral_norm


@DISCRIMINATORS.register()
class AnimeDiscriminator(nn.Layer):
    def __init__(self, channel: int = 64, nblocks: int = 3) -> None:
        super().__init__()
        channel = channel // 2
        last_channel = channel
        f = [
            spectral_norm(
                nn.Conv2D(3, channel, 3, stride=1, padding=1, bias_attr=False)),
            nn.LeakyReLU(0.2)
        ]
        in_h = 256
        for i in range(1, nblocks):
            f.extend([
                spectral_norm(
                    nn.Conv2D(last_channel,
                              channel * 2,
                              3,
                              stride=2,
                              padding=1,
                              bias_attr=False)),
                nn.LeakyReLU(0.2),
                spectral_norm(
                    nn.Conv2D(channel * 2,
                              channel * 4,
                              3,
                              stride=1,
                              padding=1,
                              bias_attr=False)),
                nn.GroupNorm(1, channel * 4),
                nn.LeakyReLU(0.2)
            ])
            last_channel = channel * 4
            channel = channel * 2
            in_h = in_h // 2

        self.body = nn.Sequential(*f)

        self.head = nn.Sequential(*[
            spectral_norm(
                nn.Conv2D(last_channel,
                          channel * 2,
                          3,
                          stride=1,
                          padding=1,
                          bias_attr=False)),
            nn.GroupNorm(1, channel * 2),
            nn.LeakyReLU(0.2),
            spectral_norm(
                nn.Conv2D(
                    channel * 2, 1, 3, stride=1, padding=1, bias_attr=False))
        ])

    def forward(self, x):
        x = self.body(x)
        x = self.head(x)
        return x
