# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import paddle.vision.transforms as T
import cv2


def get_makeup_transform(cfg, pic="image"):
    if pic == "image":
        transform = T.Compose([
            T.Resize(size=cfg.trans_size),
            T.Transpose(),
        ])
    else:
        transform = T.Resize(size=cfg.trans_size,
                             interpolation=cv2.INTER_NEAREST)

    return transform
