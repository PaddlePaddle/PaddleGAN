#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import cv2
import os.path
import numpy as np
import paddle
from .base_dataset import BaseDataset
from .image_folder import ImageFolder

from .builder import DATASETS
from .preprocess.builder import build_transforms


@DATASETS.register()
class AnimeGANV2Dataset(paddle.io.Dataset):
    """
    """
    def __init__(self,
                 dataroot,
                 style,
                 transform_real=None,
                 transform_anime=None,
                 transform_gray=None):
        """Initialize this dataset class.

        Args:
            dataroot (dict): Directory of dataset.

        """
        self.root = dataroot
        self.style = style

        self.transform_real = build_transforms(transform_real)
        self.transform_anime = build_transforms(transform_anime)
        self.transform_gray = build_transforms(transform_gray)

        self.real_root = os.path.join(self.root, 'train_photo')
        self.anime_root = os.path.join(self.root, f'{self.style}', 'style')
        self.smooth_root = os.path.join(self.root, f'{self.style}', 'smooth')

        self.real = ImageFolder(self.real_root,
                                transform=self.transform_real,
                                loader=self.loader)
        self.anime = ImageFolder(self.anime_root,
                                 transform=self.transform_anime,
                                 loader=self.loader)
        self.anime_gray = ImageFolder(self.anime_root,
                                      transform=self.transform_gray,
                                      loader=self.loader)
        self.smooth_gray = ImageFolder(self.smooth_root,
                                       transform=self.transform_gray,
                                       loader=self.loader)
        self.sizes = [
            len(fold) for fold in [self.real, self.anime, self.smooth_gray]
        ]
        self.size = max(self.sizes)
        self.reshuffle()

    @staticmethod
    def loader(path):
        return cv2.cvtColor(cv2.imread(path, flags=cv2.IMREAD_COLOR),
                            cv2.COLOR_BGR2RGB)

    def reshuffle(self):
        indexs = []
        for cur_size in self.sizes:
            x = np.arange(0, cur_size)
            np.random.shuffle(x)
            if cur_size != self.size:
                pad_num = self.size - cur_size
                pad = np.random.choice(cur_size, pad_num, replace=True)
                x = np.concatenate((x, pad))
                np.random.shuffle(x)
            indexs.append(x.tolist())
        self.indexs = list(zip(*indexs))

    def __getitem__(self, index):
        try:
            index = self.indexs.pop()
        except IndexError as e:
            self.reshuffle()
            index = self.indexs.pop()

        real_idx, anime_idx, smooth_idx = index

        return {
            'real': self.real[real_idx],
            'anime': self.anime[anime_idx],
            'anime_gray': self.anime_gray[anime_idx],
            'smooth_gray': self.smooth_gray[smooth_idx]
        }

    def __len__(self):
        return self.size
