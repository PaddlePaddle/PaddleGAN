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
import random

from .builder import DATASETS

logger = logging.getLogger(__name__)


def data_transform(img, resize_w, resize_h, load_size=286, pos=[0, 0, 256, 256], flip=True, is_image=True):
    if is_image:
        resized = img.resize((resize_w, resize_h), Image.BICUBIC)
    else:
        resized = img.resize((resize_w, resize_h), Image.NEAREST)
    croped = resized.crop((pos[0], pos[1], pos[2], pos[3]))
    fliped = ImageOps.mirror(croped) if flip else croped
    fliped = np.array(fliped) # transform to numpy array
    expanded = np.expand_dims(fliped, 2) if len(fliped.shape) < 3 else fliped
    transposed = np.transpose(expanded, (2, 0, 1)).astype('float32')
    if is_image:
        normalized = transposed / 255. * 2. - 1.
    else:
        normalized = transposed
    return normalized


@DATASETS.register()
class PhotoPenDataset(Dataset):
    def __init__(self, content_root, load_size, crop_size):
        super(PhotoPenDataset, self).__init__()
        inst_dir = os.path.join(content_root, 'train_inst')
        _, _, inst_list = next(os.walk(inst_dir))
        self.inst_list = np.sort(inst_list)
        self.content_root = content_root
        self.load_size = load_size
        self.crop_size = crop_size

    def __getitem__(self, idx):
        ins = Image.open(os.path.join(self.content_root, 'train_inst', self.inst_list[idx]))
        img = Image.open(os.path.join(self.content_root, 'train_img', self.inst_list[idx].replace(".png", ".jpg")))
        img = img.convert('RGB')

        w, h = img.size
        resize_w, resize_h = 0, 0
        if w < h:
            resize_w, resize_h = self.load_size, int(h * self.load_size / w)
        else:
            resize_w, resize_h = int(w * self.load_size / h), self.load_size
        left = random.randint(0, resize_w - self.crop_size)
        top = random.randint(0, resize_h - self.crop_size)
        flip = False
        
        img = data_transform(img, resize_w, resize_h, load_size=self.load_size, 
            pos=[left, top, left + self.crop_size, top + self.crop_size], flip=flip, is_image=True)
        ins = data_transform(ins, resize_w, resize_h, load_size=self.load_size, 
            pos=[left, top, left + self.crop_size, top + self.crop_size], flip=flip, is_image=False)
        return {'img': img, 'ins': ins, 'img_path': self.inst_list[idx]}

    def __len__(self):
        return len(self.inst_list)
    
    def name(self):
        return 'PhotoPenDataset'

@DATASETS.register()
class PhotoPenDataset_test(Dataset):
    def __init__(self, content_root, load_size, crop_size):
        super(PhotoPenDataset_test, self).__init__()
        inst_dir = os.path.join(content_root, 'test_inst')
        _, _, inst_list = next(os.walk(inst_dir))
        self.inst_list = np.sort(inst_list)
        self.content_root = content_root
        self.load_size = load_size
        self.crop_size = crop_size

    def __getitem__(self, idx):
        ins = Image.open(os.path.join(self.content_root, 'test_inst', self.inst_list[idx]))

        w, h = ins.size
        resize_w, resize_h = 0, 0
        if w < h:
            resize_w, resize_h = self.load_size, int(h * self.load_size / w)
        else:
            resize_w, resize_h = int(w * self.load_size / h), self.load_size
        left = random.randint(0, resize_w - self.crop_size)
        top = random.randint(0, resize_h - self.crop_size)
        flip = False
        
        ins = data_transform(ins, resize_w, resize_h, load_size=self.load_size, 
            pos=[left, top, left + self.crop_size, top + self.crop_size], flip=flip, is_image=False)
        return {'ins': ins, 'img_path': self.inst_list[idx]}

    def __len__(self):
        return len(self.inst_list)
    
    def name(self):
        return 'PhotoPenDataset'
