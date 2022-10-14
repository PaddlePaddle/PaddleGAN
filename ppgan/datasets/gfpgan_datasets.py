# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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
import math
import numpy as np
import random
import os

import paddle
import paddle.nn.functional as F
from paddle.vision.transforms.functional import normalize

from .builder import DATASETS

from ppgan.utils.download import get_path_from_url
from ppgan.utils.gfpgan_tools import *


@DATASETS.register()
class FFHQDegradationDataset(paddle.io.Dataset):
    """FFHQ dataset for GFPGAN.

    It reads high resolution images, and then generate low-quality (LQ) images on-the-fly.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            io_backend (dict): IO backend type and other kwarg.
            mean (list | tuple): Image mean.
            std (list | tuple): Image std.
            use_hflip (bool): Whether to horizontally flip.
            Please see more options in the codes.
    """
    def __init__(self, **opt):
        super(FFHQDegradationDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.gt_folder = opt['dataroot_gt']
        self.mean = opt['mean']
        self.std = opt['std']
        self.out_size = opt['out_size']
        self.crop_components = opt.get('crop_components', False)
        self.eye_enlarge_ratio = opt.get('eye_enlarge_ratio', 1)
        if self.crop_components:
            self.components_list = get_path_from_url(opt.get('component_path'))
            self.components_list = paddle.load(self.components_list)
            # print(self.components_list)
        self.paths = paths_from_folder(self.gt_folder)
        self.blur_kernel_size = opt['blur_kernel_size']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']
        self.blur_sigma = opt['blur_sigma']
        self.downsample_range = opt['downsample_range']
        self.noise_range = opt['noise_range']
        self.jpeg_range = opt['jpeg_range']
        self.color_jitter_prob = opt.get('color_jitter_prob')
        self.color_jitter_pt_prob = opt.get('color_jitter_pt_prob')
        self.color_jitter_shift = opt.get('color_jitter_shift', 20)
        self.gray_prob = opt.get('gray_prob')
        self.color_jitter_shift /= 255.0

    @staticmethod
    def color_jitter(img, shift):
        """jitter color: randomly jitter the RGB values, in numpy formats"""
        jitter_val = np.random.uniform(-shift, shift, 3).astype(np.float32)
        img = img + jitter_val
        img = np.clip(img, 0, 1)
        return img

    @staticmethod
    def color_jitter_pt(img, brightness, contrast, saturation, hue):
        """jitter color: randomly jitter the brightness, contrast, saturation, and hue, in torch Tensor formats"""
        fn_idx = paddle.randperm(4)
        img = paddle.to_tensor(img, dtype=img.dtype)
        for fn_id in fn_idx:
            # print('fn_id',fn_id)
            if fn_id == 0 and brightness is not None:
                brightness_factor = paddle.to_tensor(1.0).uniform_(
                    brightness[0], brightness[1]).item()
                # print("brightness_factor",brightness_factor)
                img = adjust_brightness(img, brightness_factor)
            if fn_id == 1 and contrast is not None:
                contrast_factor = paddle.to_tensor(1.0).uniform_(
                    contrast[0], contrast[1]).item()
                img = adjust_contrast(img, contrast_factor)
            if fn_id == 2 and saturation is not None:
                saturation_factor = paddle.to_tensor(1.0).uniform_(
                    saturation[0], saturation[1]).item()
                img = adjust_saturation(img, saturation_factor)
            if fn_id == 3 and hue is not None:
                hue_factor = paddle.to_tensor(1.0).uniform_(hue[0],
                                                            hue[1]).item()
                img = adjust_hue(img, hue_factor)
        return img

    def get_component_coordinates(self, index, status):
        """Get facial component (left_eye, right_eye, mouth) coordinates from a pre-loaded pth file"""
        # print(f'{index:08d}',type(self.components_list))
        components_bbox = self.components_list[f'{index:08d}']
        if status[0]:
            tmp = components_bbox['left_eye']
            components_bbox['left_eye'] = components_bbox['right_eye']
            components_bbox['right_eye'] = tmp
            components_bbox['left_eye'][
                0] = self.out_size - components_bbox['left_eye'][0]
            components_bbox['right_eye'][
                0] = self.out_size - components_bbox['right_eye'][0]
            components_bbox['mouth'][
                0] = self.out_size - components_bbox['mouth'][0]
        locations = []
        for part in ['left_eye', 'right_eye', 'mouth']:
            mean = components_bbox[part][0:2]
            half_len = components_bbox[part][2]
            if 'eye' in part:
                half_len *= self.eye_enlarge_ratio
            loc = np.hstack((mean - half_len + 1, mean + half_len))
            loc = paddle.to_tensor(loc)
            locations.append(loc)
        return locations

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'),
                                          **self.io_backend_opt)
        gt_path = self.paths[index]
        img_bytes = self.file_client.get(gt_path)
        img_gt = imfrombytes(img_bytes, float32=True)
        img_gt = cv2.resize(img_gt, (self.out_size, self.out_size))
        img_gt, status = augment(img_gt,
                                 hflip=self.opt['use_hflip'],
                                 rotation=False,
                                 return_status=True)
        h, w, _ = img_gt.shape
        if self.crop_components:
            locations = self.get_component_coordinates(index, status)
            loc_left_eye, loc_right_eye, loc_mouth = locations
        kernel = random_mixed_kernels(self.kernel_list,
                                      self.kernel_prob,
                                      self.blur_kernel_size,
                                      self.blur_sigma,
                                      self.blur_sigma, [-math.pi, math.pi],
                                      noise_range=None)
        img_lq = cv2.filter2D(img_gt, -1, kernel)
        scale = np.random.uniform(self.downsample_range[0],
                                  self.downsample_range[1])
        img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)),
                            interpolation=cv2.INTER_LINEAR)
        if self.noise_range is not None:
            img_lq = random_add_gaussian_noise(img_lq, self.noise_range)
        if self.jpeg_range is not None:
            img_lq = random_add_jpg_compression(img_lq, self.jpeg_range)
        img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)
        if self.color_jitter_prob is not None and np.random.uniform(
        ) < self.color_jitter_prob:
            img_lq = self.color_jitter(img_lq, self.color_jitter_shift)
        if self.gray_prob and np.random.uniform() < self.gray_prob:
            img_lq = cv2.cvtColor(img_lq, cv2.COLOR_BGR2GRAY)
            img_lq = np.tile(img_lq[:, :, None], [1, 1, 3])
            if self.opt.get('gt_gray'):
                img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
                img_gt = np.tile(img_gt[:, :, None], [1, 1, 3])
        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)
        if self.color_jitter_pt_prob is not None and np.random.uniform(
        ) < self.color_jitter_pt_prob:
            brightness = self.opt.get('brightness', (0.5, 1.5))
            contrast = self.opt.get('contrast', (0.5, 1.5))
            saturation = self.opt.get('saturation', (0, 1.5))
            hue = self.opt.get('hue', (-0.1, 0.1))
            img_lq = self.color_jitter_pt(img_lq, brightness, contrast,
                                          saturation, hue)
        img_lq = np.clip((img_lq * 255.0).round(), 0, 255) / 255.0
        img_gt = normalize(img_gt, self.mean, self.std)
        img_lq = normalize(img_lq, self.mean, self.std)
        if self.crop_components:
            return_dict = {
                'lq': img_lq,
                'gt': img_gt,
                'gt_path': gt_path,
                'loc_left_eye': loc_left_eye,
                'loc_right_eye': loc_right_eye,
                'loc_mouth': loc_mouth
            }
            return return_dict
        else:
            return {'lq': img_lq, 'gt': img_gt, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)
