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
from paddle.vision.transforms.transforms import _check_input

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
class PairedRandomRotation(T.RandomRotation):
    def __init__(self, degrees, resample=False, expand=False, center=None, fill=0, keys=None):
        super().__init__(degrees, resample, expand, center, fill, keys=keys)

    def _get_params(self, input):
        angle = random.uniform(self.degrees[0], self.degrees[1])
        return angle

    def _apply_image(self, img):
        """
        Args:
            img (PIL.Image|np.array): Image to be rotated.
        Returns:
            PIL.Image or np.array: Rotated image.
        """
        angle = self.params
        return F.rotate(img, angle, self.resample, self.expand, self.center,
                        self.fill)


@TRANSFORMS.register()
class PairedRandomResizedCrop(T.RandomResizedCrop):
    def __init__(self,
                 size,
                 scale=(0.08, 1.0),
                 ratio=(3. / 4, 4. / 3),
                 interpolation='bilinear',
                 keys=None):
        super().__init__(size, scale, ratio, interpolation, keys=keys)
        self.param = None
    
    def _apply_image(self, img):
        if self.param is None:
            self.param = self._get_param(img)
        
        i, j, h, w = self.param
        cropped_img = F.crop(img, i, j, h, w)
        return F.resize(cropped_img, self.size, self.interpolation)


@TRANSFORMS.register()
class PairedBrightnessTransform(T.BrightnessTransform):
    def __init__(self, value, keys=None):
        super().__init__(value, keys=keys)
        
    def _get_params(self, input):
        return random.uniform(self.value[0], self.value[1]) if self.value is not None else None
    
    def _apply_image(self, img):
        if self.value is None:
            return img
        return F.adjust_brightness(img, self.params)


@TRANSFORMS.register()
class PairedContrastTransform(T.ContrastTransform):
    def __init__(self, value, keys=None):
        super().__init__(value, keys=keys)
    
    def _get_params(self, input):
        return random.uniform(self.value[0], self.value[1]) if self.value is not None else None
    
    def _apply_image(self, img):
        if self.value is None:
            return img
        return F.adjust_contrast(img, self.params)


@TRANSFORMS.register()
class PairedSaturationTransform(T.SaturationTransform):
    def __init__(self, value, keys=None):
        super().__init__(value, keys=keys)
    
    def _get_params(self, input):
        return random.uniform(self.value[0], self.value[1]) if self.value is not None else None
    
    def _apply_image(self, img):
        if self.value is None:
            return img
        return F.adjust_saturation(img, self.params)


@TRANSFORMS.register()
class PairedHueTransform(T.HueTransform):
    def __init__(self, value, keys=None):
        super().__init__(value, keys=keys)
    
    def _get_params(self, input):
        return random.uniform(self.value[0], self.value[1]) if self.value is not None else None
    
    def _apply_image(self, img):
        if self.value is None:
            return img
        return F.adjust_hue(img, self.params)


@TRANSFORMS.register()
class PairedColorJitter(T.BaseTransform):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, keys=None):
        super().__init__(keys=keys)
        self.brightness = _check_input(brightness, 'brightness')
        self.contrast = _check_input(contrast, 'contrast')
        self.saturation = _check_input(saturation, 'saturation')
        self.hue = _check_input(hue, 'hue', center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)
    
    def _get_params(self, input):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []
        
        if self.brightness is not None:
            brightness = random.uniform(self.brightness[0], self.brightness[1])
            f = lambda img: F.adjust_brightness(img, brightness)
            transforms.append(f)

        if self.contrast is not None:
            contrast = random.uniform(self.contrast[0], self.contrast[1])
            f = lambda img: F.adjust_contrast(img, contrast)
            transforms.append(f)

        if self.saturation is not None:
            saturation = random.uniform(self.saturation[0], self.saturation[1])
            f = lambda img: F.adjust_saturation(img, saturation)
            transforms.append(f)

        if self.hue is not None:
            hue = random.uniform(self.hue[0], self.hue[1])
            f = lambda img: F.adjust_hue(img, hue)
            transforms.append(f)

        random.shuffle(transforms)
        return transforms
    
    def _apply_image(self, img):
        for f in self.params:
            img = f(img)
        return img