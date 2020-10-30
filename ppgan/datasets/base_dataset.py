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

# code was heavily based on https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
import random
import numpy as np

from paddle.io import Dataset
from PIL import Image
import cv2

import paddle.vision.transforms as transforms
from .transforms import transforms as T
from abc import ABC, abstractmethod


class BaseDataset(Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.
    """
    def __init__(self, cfg):
        """Initialize the class; save the options in the class

        Args:
            cfg (dict) -- stores all the experiment flags
        """
        self.cfg = cfg
        self.root = cfg.dataroot

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass


def get_params(cfg, size):
    w, h = size
    new_h = h
    new_w = w
    if cfg.preprocess == 'resize_and_crop':
        new_h = new_w = cfg.load_size
    elif cfg.preprocess == 'scale_width_and_crop':
        new_w = cfg.load_size
        new_h = cfg.load_size * h // w

    x = random.randint(0, np.maximum(0, new_w - cfg.crop_size))
    y = random.randint(0, np.maximum(0, new_h - cfg.crop_size))

    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}


def get_transform(cfg,
                  params=None,
                  grayscale=False,
                  method=cv2.INTER_CUBIC,
                  convert=True):
    transform_list = []
    if grayscale:
        print('grayscale not support for now!!!')
        pass
    if 'resize' in cfg.preprocess:
        osize = (cfg.load_size, cfg.load_size)
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in cfg.preprocess:
        print('scale_width not support for now!!!')
        pass

    if 'crop' in cfg.preprocess:

        if params is None:
            transform_list.append(T.RandomCrop(cfg.crop_size))
        else:
            transform_list.append(T.Crop(params['crop_pos'], cfg.crop_size))

    if cfg.preprocess == 'none':
        print('preprocess not support for now!!!')
        pass

    if not cfg.no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.RandomHorizontalFlip(1.0))

    if convert:
        transform_list += [transforms.Permute(to_rgb=True)]
        if cfg.get('normalize', None):
            transform_list += [
                transforms.Normalize(cfg.normalize.mean, cfg.normalize.std)
            ]

    return transforms.Compose(transform_list)
