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

import paddle
import paddle.nn as nn

from .builder import DISCRIMINATORS


@DISCRIMINATORS.register()
class LapStyleDiscriminator(nn.Layer):
    def __init__(self):
        super(LapStyleDiscriminator, self).__init__()
        num_layer = 3
        num_channel = 32
        self.head = nn.Sequential(
            ('conv',
             nn.Conv2D(3, num_channel, kernel_size=3, stride=1, padding=1)),
            ('norm', nn.BatchNorm2D(num_channel)),
            ('LeakyRelu', nn.LeakyReLU(0.2)))
        self.body = nn.Sequential()
        for i in range(num_layer - 2):
            self.body.add_sublayer(
                'conv%d' % (i + 1),
                nn.Conv2D(num_channel,
                          num_channel,
                          kernel_size=3,
                          stride=1,
                          padding=1))
            self.body.add_sublayer('norm%d' % (i + 1),
                                   nn.BatchNorm2D(num_channel))
            self.body.add_sublayer('LeakyRelu%d' % (i + 1), nn.LeakyReLU(0.2))
        self.tail = nn.Conv2D(num_channel,
                              1,
                              kernel_size=3,
                              stride=1,
                              padding=1)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x
