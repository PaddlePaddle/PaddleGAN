# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.utils.download import get_weights_path_from_url
from paddle.vision.models.vgg import make_layers

cfg = [
    64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512,
    512, 512, 'M'
]

model_urls = {
    'vgg16': ('https://paddle-hapi.bj.bcebos.com/models/vgg16.pdparams',
              '89bbffc0f87d260be9b8cdc169c991c4')
}


class VGG(nn.Layer):
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features

    def forward(self, x):
        x = self.features(x)
        return x


def vgg16(pretrained=False):
    features = make_layers(cfg)
    model = VGG(features)

    if pretrained:
        weight_path = get_weights_path_from_url(model_urls['vgg16'][0],
                                                model_urls['vgg16'][1])
        param, _ = paddle.load(weight_path)
        model.load_dict(param)

    return model
