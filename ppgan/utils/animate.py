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

# code was heavily based on https://github.com/AliaksandrSiarohin/first-order-model

import numpy as np
from scipy.spatial import ConvexHull

import paddle


def normalize_kp(kp_source,
                 kp_driving,
                 kp_driving_initial,
                 adapt_movement_scale=False,
                 use_relative_movement=False,
                 use_relative_jacobian=False):
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source['value'][0].numpy()).volume
        driving_area = ConvexHull(kp_driving_initial['value'][0].numpy()).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    kp_new = {k: v for k, v in kp_driving.items()}

    if use_relative_movement:
        kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
        kp_value_diff *= adapt_movement_scale
        kp_new['value'] = kp_value_diff + kp_source['value']

        if use_relative_jacobian:
            jacobian_diff = paddle.matmul(
                kp_driving['jacobian'],
                paddle.inverse(kp_driving_initial['jacobian']))
            kp_new['jacobian'] = paddle.matmul(jacobian_diff,
                                               kp_source['jacobian'])

    return kp_new
