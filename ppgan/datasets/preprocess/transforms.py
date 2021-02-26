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
import cv2
import glob
import random
import numbers
import collections
import numpy as np

from PIL import Image

import paddle.vision.transforms as T
import paddle.vision.transforms.functional as F

from .builder import TRANSFORMS, build_from_config
from .builder import PREPROCESS

if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable

TRANSFORMS.register(T.Resize)
TRANSFORMS.register(T.RandomCrop)
TRANSFORMS.register(T.RandomHorizontalFlip)
TRANSFORMS.register(T.RandomVerticalFlip)
TRANSFORMS.register(T.Normalize)
TRANSFORMS.register(T.Transpose)
TRANSFORMS.register(T.Grayscale)


@PREPROCESS.register()
class Transforms():
    def __init__(self, pipeline, input_keys, output_keys=None):
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.transforms = []
        for transform_cfg in pipeline:
            self.transforms.append(build_from_config(transform_cfg, TRANSFORMS))

    def __call__(self, datas):
        data = []
        for k in self.input_keys:
            data.append(datas[k])
        data = tuple(data)
        for transform in self.transforms:
            data = transform(data)
            if hasattr(transform, 'params') and isinstance(
                    transform.params, dict):
                datas.update(transform.params)

        if len(self.input_keys) > 1:
            for i, k in enumerate(self.input_keys):
                datas[k] = data[i]
        else:
            datas[k] = data

        if self.output_keys is not None:
            for i, k in enumerate(self.output_keys):
                datas[k] = data[i]
            return datas

        return datas


@PREPROCESS.register()
class SplitPairedImage:
    def __init__(self, key, paired_keys=['A', 'B']):
        self.key = key
        self.paired_keys = paired_keys

    def __call__(self, datas):
        # split AB image into A and B
        h, w = datas[self.key].shape[:2]
        # w, h = AB.size
        w2 = int(w / 2)

        a, b = self.paired_keys
        datas[a] = datas[self.key][:h, :w2, :]
        datas[b] = datas[self.key][:h, w2:, :]

        datas[a + '_path'] = datas[self.key + '_path']
        datas[b + '_path'] = datas[self.key + '_path']

        return datas


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
class PairedRandomVerticalFlip(T.RandomHorizontalFlip):
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
class PairedRandomTransposeHW(T.BaseTransform):
    """Randomly transpose images in H and W dimensions with a probability.

    (TransposeHW = horizontal flip + anti-clockwise rotatation by 90 degrees)
    When used with horizontal/vertical flips, it serves as a way of rotation
    augmentation.
    It also supports randomly transposing a list of images.

    Required keys are the keys in attributes "keys", added or modified keys are
    "transpose" and the keys in attributes "keys".

    Args:
        prob (float): The propability to transpose the images.
        keys (list[str]): The images to be transposed.
    """
    def __init__(self, prob=0.5, keys=None):
        self.keys = keys
        self.prob = prob

    def _get_params(self, inputs):
        params = {}
        params['transpose'] = random.random() < self.prob
        return params

    def _apply_image(self, image):
        if self.params['transpose']:
            image = image.transpose(1, 0, 2)
        return image


@TRANSFORMS.register()
class SRPairedRandomCrop(T.BaseTransform):
    """Super resolution random crop.

    It crops a pair of lq and gt images with corresponding locations.
    It also supports accepting lq list and gt list.
    Required keys are "scale", "lq", and "gt",
    added or modified keys are "lq" and "gt".

    Args:
        scale (int): model upscale factor.
        gt_patch_size (int): cropped gt patch size.
    """
    def __init__(self, scale, gt_patch_size, scale_list=False, keys=None):
        self.gt_patch_size = gt_patch_size
        self.scale = scale
        self.keys = keys
        self.scale_list = scale_list

    def __call__(self, inputs):
        """inputs must be (lq_img, gt_img)"""
        scale = self.scale
        lq_patch_size = self.gt_patch_size // scale

        lq = inputs[0]
        gt = inputs[1]

        h_lq, w_lq, _ = lq.shape
        h_gt, w_gt, _ = gt.shape

        if h_gt != h_lq * scale or w_gt != w_lq * scale:
            raise ValueError('scale size not match')
        if h_lq < lq_patch_size or w_lq < lq_patch_size:
            raise ValueError('lq size error')

        # randomly choose top and left coordinates for lq patch
        top = random.randint(0, h_lq - lq_patch_size)
        left = random.randint(0, w_lq - lq_patch_size)
        # crop lq patch
        lq = lq[top:top + lq_patch_size, left:left + lq_patch_size, ...]
        # crop corresponding gt patch
        top_gt, left_gt = int(top * scale), int(left * scale)
        gt = gt[top_gt:top_gt + self.gt_patch_size,
                left_gt:left_gt + self.gt_patch_size, ...]

        if self.scale_list and self.scale == 4:
            lqx2 = F.resize(gt, (lq_patch_size * 2, lq_patch_size * 2),
                            'bicubic')
            outputs = (lq, lqx2, gt)
            return outputs

        outputs = (lq, gt)
        return outputs


@TRANSFORMS.register()
class SRNoise(T.BaseTransform):
    """Super resolution noise.

    Args:
        noise_path (str): directory of noise image.
        size (int): cropped noise patch size.
    """
    def __init__(self, noise_path, size, keys=None):
        self.noise_path = noise_path
        self.noise_imgs = sorted(glob.glob(noise_path + '*.png'))
        self.size = size
        self.keys = keys
        self.transform = T.Compose([
            T.RandomCrop(size),
            T.Transpose(),
            T.Normalize([0., 0., 0.], [255., 255., 255.])
        ])

    def _apply_image(self, image):
        idx = np.random.randint(0, len(self.noise_imgs))
        noise = self.transform(Image.open(self.noise_imgs[idx]))
        normed_noise = noise - np.mean(noise, axis=(1, 2), keepdims=True)
        image = image + normed_noise
        image = np.clip(image, 0., 1.)
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
        return np.clip(image + self.params['value'], 0, 255).astype('uint8')
        # return custom_F.add(image, self.params['value'])


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
