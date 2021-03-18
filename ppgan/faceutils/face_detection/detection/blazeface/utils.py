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

import cv2
import numpy as np


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)

    return resized


def resize_and_crop_image(image, dim):
    if image.shape[0] > image.shape[1]:
        img = image_resize(image, width=dim)
        yshift, xshift = (image.shape[0] - image.shape[1]) // 2, 0
        y_start = (img.shape[0] - img.shape[1]) // 2
        y_end = y_start + dim
        return img[y_start:y_end, :, :], (xshift, yshift)
    else:
        img = image_resize(image, height=dim)
        yshift, xshift = 0, (image.shape[1] - image.shape[0]) // 2
        x_start = (img.shape[1] - img.shape[0]) // 2
        x_end = x_start + dim
        return img[:, x_start:x_end, :], (xshift, yshift)


def resize_and_crop_batch(frames, dim):
    smframes = []
    xshift, yshift = 0, 0
    for i in range(len(frames)):
        smframe, (xshift, yshift) = resize_and_crop_image(frames[i], dim)
        smframes.append(smframe)
    smframes = np.stack(smframes)
    return smframes, (xshift, yshift)
