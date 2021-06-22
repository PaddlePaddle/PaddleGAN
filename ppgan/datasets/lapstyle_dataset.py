# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

import logging
import os
import numpy as np
from PIL import Image
import paddle
import paddle.vision.transforms as T
from paddle.io import Dataset
import cv2

from .builder import DATASETS

logger = logging.getLogger(__name__)


def data_transform(crop_size):
    transform_list = [T.RandomCrop(crop_size)]
    return T.Compose(transform_list)


@DATASETS.register()
class LapStyleDataset(Dataset):
    """
    coco2017 dataset for LapStyle model
    """
    def __init__(self, content_root, style_root, load_size, crop_size):
        super(LapStyleDataset, self).__init__()
        self.content_root = content_root
        self.paths = os.listdir(self.content_root)
        self.style_root = style_root
        self.load_size = load_size
        self.crop_size = crop_size
        self.transform = data_transform(self.crop_size)

    def __getitem__(self, index):
        """Get training sample

        return:
            ci: content image with shape [C,W,H],
            si: style image with shape [C,W,H],
            ci_path: str
        """
        path = self.paths[index]
        content_img = cv2.imread(os.path.join(self.content_root, path))
        if content_img.ndim == 2:
            content_img = cv2.cvtColor(content_img, cv2.COLOR_GRAY2RGB)
        else:
            content_img = cv2.cvtColor(content_img, cv2.COLOR_BGR2RGB)
        content_img = Image.fromarray(content_img)
        content_img = content_img.resize((self.load_size, self.load_size),
                                         Image.BILINEAR)
        content_img = np.array(content_img)
        style_img = cv2.imread(self.style_root)
        style_img = cv2.cvtColor(style_img, cv2.COLOR_BGR2RGB)
        style_img = Image.fromarray(style_img)
        style_img = style_img.resize((self.load_size, self.load_size),
                                     Image.BILINEAR)
        style_img = np.array(style_img)
        content_img = self.transform(content_img)
        style_img = self.transform(style_img)
        content_img = self.img(content_img)
        style_img = self.img(style_img)
        return {'ci': content_img, 'si': style_img, 'ci_path': path}

    def img(self, img):
        """make image with [0,255] and HWC to [0,1] and CHW

        return:
            img: image with shape [3,W,H] and value [0, 1].
        """
        # [0,255] to [0,1]
        img = img.astype(np.float32) / 255.
        # some images have 4 channels
        if img.shape[2] > 3:
            img = img[:, :, :3]
        # HWC to CHW
        img = np.transpose(img, (2, 0, 1)).astype('float32')
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'LapStyleDataset'
