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

import sys
import random
import numbers
import collections
import numpy as np

import paddle.vision.transforms as T
import paddle.vision.transforms.functional as F

from . import functional as custom_F
from .builder import TRANSFORMS

if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable

TRANSFORMS.register(T.Resize)
TRANSFORMS.register(T.RandomCrop)
TRANSFORMS.register(T.RandomHorizontalFlip)
TRANSFORMS.register(T.Normalize)
TRANSFORMS.register(T.Transpose)
TRANSFORMS.register(T.Grayscale)


@TRANSFORMS.register()
class PairedRandomCrop(T.RandomCrop):
    def __init__(self, size, keys=None):
        super().__init__(size, keys=keys)

        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def _get_params(self, inputs):
        image = inputs[self.keys.index('image')]
        params = {}
        params['crop_prams'] = self._get_param(image, self.size)
        return params

    def _apply_image(self, img):
        i, j, h, w = self.params['crop_prams']
        return F.crop(img, i, j, h, w)


@TRANSFORMS.register()
class PairedRandomHorizontalFlip(T.RandomHorizontalFlip):
    def __init__(self, prob=0.5, keys=None):
        super().__init__(prob, keys=keys)

    def _get_params(self, inputs):
        params = {}
        params['flip'] = random.random() < self.prob
        return params

    def _apply_image(self, image):
        if self.params['flip']:
            return F.hflip(image)
        return image


@TRANSFORMS.register()
class Add(T.BaseTransform):
    def __init__(self, value, keys=None):
        """Initialize Add Transform

        Parameters:
            value (List[int]) -- the [r,g,b] value will add to image by pixel wise.
        """
        super().__init__(keys=keys)
        self.value = value

    def _get_params(self, inputs):
        params = {}
        params['value'] = self.value
        return params

    def _apply_image(self, image):
        return custom_F.add(image, self.params['value'])


@TRANSFORMS.register()
class ResizeToScale(T.BaseTransform):
    def __init__(self,
                 size: int,
                 scale: int,
                 interpolation='bilinear',
                 keys=None):
        """Initialize ResizeToScale Transform

        Parameters:
            size (List[int]) -- the minimum target size
            scale (List[int]) -- the stride scale
            interpolation (Optional[str]) -- interpolation method
        """
        super().__init__(keys=keys)
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.scale = scale
        self.interpolation = interpolation

    def _get_params(self, inputs):
        image = inputs[self.keys.index('image')]
        hw = image.shape[:2]
        params = {}
        params['taget_size'] = self.reduce_to_scale(hw, self.size[::-1],
                                                    self.scale)
        return params

    @staticmethod
    def reduce_to_scale(img_hw, min_hw, scale):
        im_h, im_w = img_hw
        if im_h <= min_hw[0]:
            im_h = min_hw[0]
        else:
            x = im_h % scale
            im_h = im_h - x

        if im_w < min_hw[1]:
            im_w = min_hw[1]
        else:
            y = im_w % scale
            im_w = im_w - y
        return (im_h, im_w)

    def _apply_image(self, image):
        return F.resize(image, self.params['taget_size'], self.interpolation)
