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

from PIL import Image, ImageOps
import os
import numpy as np
import logging

from paddle.io import Dataset, DataLoader
from paddle.vision.transforms import Compose, RandomResizedCrop, RandomHorizontalFlip, RandomRotation, ColorJitter, Resize

from .builder import DATASETS

logger = logging.getLogger(__name__)

@DATASETS.register()
class AOTGANDataset(Dataset):
    def __init__(self, dataset_path, img_size, istrain=True):
        super(AOTGANDataset, self).__init__()

        self.image_path = []
        def get_all_sub_dirs(root_dir): # read all image files including subdirectories
            file_list = []
            def get_sub_dirs(r_dir):
                for root, dirs, files in os.walk(r_dir):
                    if len(files) > 0:
                        for f in files:
                            file_list.append(os.path.join(root, f))
                    if len(dirs) > 0:
                        for d in dirs:
                            get_sub_dirs(os.path.join(root, d))
                    break
            get_sub_dirs(root_dir)
            return file_list

        # set data path
        if istrain:
            self.img_list = get_all_sub_dirs(os.path.join(dataset_path, 'train_img'))
            self.mask_dir = os.path.join(dataset_path, 'train_mask')
        else:
            self.img_list = get_all_sub_dirs(os.path.join(dataset_path, 'val_img'))
            self.mask_dir = os.path.join(dataset_path, 'val_mask')
        self.img_list = np.sort(np.array(self.img_list))
        _, _, mask_list = next(os.walk(self.mask_dir))
        self.mask_list = np.sort(mask_list)


        self.istrain = istrain

        # augumentations
        if istrain:
            self.img_trans = Compose([
                Resize(img_size),
                RandomResizedCrop(img_size),
                RandomHorizontalFlip(),
                ColorJitter(0.05, 0.05, 0.05, 0.05),
            ])
            self.mask_trans = Compose([
                Resize([img_size, img_size], interpolation='nearest'),
                RandomHorizontalFlip(),
            ])
        else:
            self.img_trans = Compose([
                Resize([img_size, img_size], interpolation='bilinear'),
            ])
            self.mask_trans = Compose([
                Resize([img_size, img_size], interpolation='nearest'),
            ])

        self.istrain = istrain

    # feed data
    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx])
        mask = Image.open(os.path.join(self.mask_dir, self.mask_list[np.random.randint(0, self.mask_list.shape[0])]))
        img = self.img_trans(img)
        mask = self.mask_trans(mask)

        mask = mask.rotate(np.random.randint(0, 45))
        img = img.convert('RGB')
        mask = mask.convert('L')

        img = np.array(img).astype('float32')
        img = (img / 255.) * 2. - 1.
        img = np.transpose(img, (2, 0, 1))
        mask = np.array(mask).astype('float32') / 255.
        mask = np.expand_dims(mask, 0)

        return {'img':img, 'mask':mask, 'img_path':self.img_list[idx]}

    def __len__(self):
        return len(self.img_list)

    def name(self):
        return 'PlaceDateset'

@DATASETS.register()
class AOTGANDataset_test(Dataset):
    def __init__(self, dataset_path, img_size, istrain=True):
        super(AOTGANDataset_test, self).__init__()

        self.image_path = []
        def get_all_sub_dirs(root_dir): # read all image files including subdirectories
            file_list = []
            def get_sub_dirs(r_dir):
                for root, dirs, files in os.walk(r_dir):
                    if len(files) > 0:
                        for f in files:
                            file_list.append(os.path.join(root, f))
                    if len(dirs) > 0:
                        for d in dirs:
                            get_sub_dirs(os.path.join(root, d))
                    break
            get_sub_dirs(root_dir)
            return file_list

        # set data path
        if istrain:
            self.img_list = get_all_sub_dirs(os.path.join(dataset_path, 'train_img'))
            self.mask_dir = os.path.join(dataset_path, 'train_mask')
        else:
            self.img_list = get_all_sub_dirs(os.path.join(dataset_path, 'val_img'))
            self. mask_dir = os.path.join(dataset_path, 'val_mask')
        self.img_list = np.sort(np.array(self.img_list))
        _, _, mask_list = next(os.walk(self.mask_dir))
        self.mask_list = np.sort(mask_list)


        self.istrain = istrain

        # augumentations
        if istrain:
            self.img_trans = Compose([
                RandomResizedCrop(img_size),
                RandomHorizontalFlip(),
                ColorJitter(0.05, 0.05, 0.05, 0.05),
            ])
            self.mask_trans = Compose([
                Resize([img_size, img_size], interpolation='nearest'),
                RandomHorizontalFlip(),
            ])
        else:
            self.img_trans = Compose([
                Resize([img_size, img_size], interpolation='bilinear'),
            ])
            self.mask_trans = Compose([
                Resize([img_size, img_size], interpolation='nearest'),
            ])

        self.istrain = istrain

    # feed data
    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx])
        mask = Image.open(os.path.join(self.mask_dir, self.mask_list[np.random.randint(0, self.mask_list.shape[0])]))
        img = self.img_trans(img)
        mask = self.mask_trans(mask)

        mask = mask.rotate(np.random.randint(0, 45))
        img = img.convert('RGB')
        mask = mask.convert('L')

        img = np.array(img).astype('float32')
        img = (img / 255.) * 2. - 1.
        img = np.transpose(img, (2, 0, 1))
        mask = np.array(mask).astype('float32') / 255.
        mask = np.expand_dims(mask, 0)

        return {'img':img, 'mask':mask, 'img_path':self.img_list[idx]}

    def __len__(self):
        return len(self.img_list)

    def name(self):
        return 'PlaceDateset_test'
