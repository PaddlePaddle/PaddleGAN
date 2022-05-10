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
from .helpers import get_blocks, Flatten, bottleneck_IR, bottleneck_IR_SE, l2_norm
"""
Modified Backbone implementation from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch)
"""


class Backbone(nn.Layer):

    def __init__(self,
                 input_size,
                 num_layers,
                 mode='ir',
                 drop_ratio=0.4,
                 affine=True):
        super(Backbone, self).__init__()
        assert input_size in [112, 224], "input_size should be 112 or 224"
        assert num_layers in [50, 100,
                              152], "num_layers should be 50, 100 or 152"
        assert mode in ['ir', 'ir_se'], "mode should be ir or ir_se"
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = paddle.nn.Sequential(
            nn.Conv2D(3, 64, (3, 3), 1, 1, bias_attr=False), nn.BatchNorm2D(64),
            nn.PReLU(64))
        if input_size == 112:
            self.output_layer = nn.Sequential(nn.BatchNorm2D(512),
                                              nn.Dropout(drop_ratio), Flatten(),
                                              nn.Linear(512 * 7 * 7, 512),
                                              nn.BatchNorm1D(512))
        else:
            self.output_layer = nn.Sequential(nn.BatchNorm2D(512),
                                              nn.Dropout(drop_ratio), Flatten(),
                                              nn.Linear(512 * 14 * 14, 512),
                                              nn.BatchNorm1D(512))

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel, bottleneck.depth,
                                bottleneck.stride))
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return l2_norm(x)
