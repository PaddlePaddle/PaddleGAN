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
import paddle.nn.functional as F
from .first_order import Hourglass, make_coordinate_grid, AntiAliasInterpolation2d


class KPDetector(nn.Layer):
    """
    Detecting a keypoints. Return keypoint position and jacobian near each keypoint.
    """
    def __init__(self,
                 block_expansion,
                 num_kp,
                 num_channels,
                 max_features,
                 num_blocks,
                 temperature,
                 estimate_jacobian=False,
                 scale_factor=1,
                 single_jacobian_map=False,
                 pad=0):
        super(KPDetector, self).__init__()

        self.predictor = Hourglass(block_expansion,
                                   in_features=num_channels,
                                   max_features=max_features,
                                   num_blocks=num_blocks)

        self.kp = nn.Conv2D(in_channels=self.predictor.out_filters,
                            out_channels=num_kp,
                            kernel_size=(7, 7),
                            padding=pad)

        if estimate_jacobian:
            self.num_jacobian_maps = 1 if single_jacobian_map else num_kp
            self.jacobian = nn.Conv2D(in_channels=self.predictor.out_filters,
                                      out_channels=4 * self.num_jacobian_maps,
                                      kernel_size=(7, 7),
                                      padding=pad)
            # self.jacobian.weight.data.zero_()
            # self.jacobian.bias.data.copy_(paddle.tensor([1, 0, 0, 1] * self.num_jacobian_maps, dtype='float32'))
        else:
            self.jacobian = None

        self.temperature = temperature
        self.scale_factor = scale_factor
        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels,
                                                 self.scale_factor)

    def gaussian2kp(self, heatmap):
        """
        Extract the mean and from a heatmap
        """
        shape = heatmap.shape
        heatmap = heatmap.unsqueeze(-1)
        grid = make_coordinate_grid(shape[2:],
                                    heatmap.dtype).unsqueeze(0).unsqueeze(0)
        value = (heatmap * grid).sum(axis=(2, 3))

        kp = {'value': value}

        return kp

    def forward(self, x):
        if self.scale_factor != 1:
            x = self.down(x)

        feature_map = self.predictor(x)
        prediction = self.kp(feature_map)

        final_shape = prediction.shape
        heatmap = prediction.reshape([final_shape[0], final_shape[1], -1])
        heatmap = F.softmax(heatmap / self.temperature, axis=2)
        heatmap = heatmap.reshape([*final_shape])

        out = self.gaussian2kp(heatmap)

        if self.jacobian is not None:
            jacobian_map = self.jacobian(feature_map)
            jacobian_map = jacobian_map.reshape([
                final_shape[0], self.num_jacobian_maps, 4, final_shape[2],
                final_shape[3]
            ])
            heatmap = heatmap.unsqueeze(2)

            jacobian = heatmap * jacobian_map
            jacobian = jacobian.reshape([final_shape[0], final_shape[1], 4, -1])
            jacobian = jacobian.sum(axis=-1)
            jacobian = jacobian.reshape(
                [jacobian.shape[0], jacobian.shape[1], 2, 2])
            out['jacobian'] = jacobian

        return out
