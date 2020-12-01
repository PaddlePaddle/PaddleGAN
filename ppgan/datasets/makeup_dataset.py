# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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
import random
import os.path
import numpy as np
from PIL import Image

import paddle
import paddle.vision.transforms as T
from .base_dataset import BaseDataset
from ..utils.preprocess import *

from .builder import DATASETS


@DATASETS.register()
class MakeupDataset(paddle.io.Dataset):
    def __init__(self, dataroot, phase, trans_size, cls_list):
        """Initialize psgan dataset class.

        Args:
            dataroot (str): Directory of dataset.
            phase (str): 'train' or 'test'.
        """
        self.image_path = dataroot
        self.mode = phase
        self.trans_size = trans_size
        self.cls_list = cls_list
        self.transform = self.build_makeup_transform()

        self.norm = T.Normalize([127.5, 127.5, 127.5], [127.5, 127.5, 127.5])
        self.transform_mask = self.build_makeup_transform("mask")
        self.trans_size = trans_size

        self.cls_A = self.cls_list[0]
        self.cls_B = self.cls_list[1]
        for cls in self.cls_list:
            setattr(
                self, cls + "_list_path",
                os.path.join(self.image_path, self.mode + '_' + cls + ".txt"))
            setattr(self, cls + "_lines",
                    open(getattr(self, cls + "_list_path"), 'r').readlines())
            setattr(self, "num_of_" + cls + "_data",
                    len(getattr(self, cls + "_lines")))
        print('Start preprocessing dataset..!')
        self.preprocess()
        print('Finished preprocessing dataset..!')

    def preprocess(self):
        """preprocess image"""
        for cls in self.cls_list:
            setattr(self, cls + "_filenames", [])
            setattr(self, cls + "_mask_filenames", [])
            setattr(self, cls + "_lmks_filenames", [])

            lines = getattr(self, cls + "_lines")
            random.shuffle(lines)

            for i, line in enumerate(lines):
                splits = line.split()
                getattr(self, cls + "_filenames").append(splits[0])
                getattr(self, cls + "_mask_filenames").append(splits[1])
                getattr(self, cls + "_lmks_filenames").append(splits[2])

    def build_makeup_transform(self, pic="image"):
        if pic == "image":
            transform = T.Compose([
                T.Resize(size=self.trans_size),
                T.Transpose(),
            ])
        else:
            transform = T.Resize(size=self.trans_size,
                                 interpolation=cv2.INTER_NEAREST)

        return transform

    def __getitem__(self, index):
        """Return MANet and MDNet needed params.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains needed params.
        """
        try:
            index_A = random.randint(
                0, getattr(self, "num_of_" + self.cls_A + "_data"))
            index_B = random.randint(
                0, getattr(self, "num_of_" + self.cls_B + "_data"))

            if self.mode == 'test':
                num_b = getattr(self, 'num_of_' + self.cls_list[1] + '_data')
                index_A = int(index / num_b)
                index_B = int(index % num_b)
            image_A = Image.open(
                os.path.join(self.image_path,
                             getattr(self, self.cls_A +
                                     "_filenames")[index_A])).convert("RGB")

            image_B = Image.open(
                os.path.join(self.image_path,
                             getattr(self, self.cls_B +
                                     "_filenames")[index_B])).convert("RGB")
            mask_A = np.array(
                Image.open(
                    os.path.join(
                        self.image_path,
                        getattr(self,
                                self.cls_A + "_mask_filenames")[index_A])))
            mask_B = np.array(
                Image.open(
                    os.path.join(
                        self.image_path,
                        getattr(self, self.cls_B +
                                "_mask_filenames")[index_B])).convert('L'))
            image_A = np.array(image_A)
            image_B = np.array(image_B)

            image_A = self.transform(image_A)
            image_B = self.transform(image_B)

            mask_A = cv2.resize(mask_A, (256, 256),
                                interpolation=cv2.INTER_NEAREST)
            mask_B = cv2.resize(mask_B, (256, 256),
                                interpolation=cv2.INTER_NEAREST)

            lmks_A = np.loadtxt(
                os.path.join(
                    self.image_path,
                    getattr(self, self.cls_A + "_lmks_filenames")[index_A]))
            lmks_B = np.loadtxt(
                os.path.join(
                    self.image_path,
                    getattr(self, self.cls_B + "_lmks_filenames")[index_B]))
            lmks_A = lmks_A / image_A.shape[:2] * self.trans_size
            lmks_B = lmks_B / image_B.shape[:2] * self.trans_size

            P_A = generate_P_from_lmks(lmks_A, self.trans_size,
                                       image_A.shape[0], image_A.shape[1])

            P_B = generate_P_from_lmks(lmks_B, self.trans_size,
                                       image_B.shape[0], image_B.shape[1])

            mask_A_aug = generate_mask_aug(mask_A, lmks_A)
            mask_B_aug = generate_mask_aug(mask_B, lmks_B)

            consis_mask = calculate_consis_mask(mask_A_aug, mask_B_aug)
            consis_mask_idt_A = calculate_consis_mask(mask_A_aug, mask_A_aug)
            consis_mask_idt_B = calculate_consis_mask(mask_A_aug, mask_B_aug)

        except Exception as e:
            print(e)
            return self.__getitem__(index + 1)
        return {
            'image_A': self.norm(image_A),
            'image_B': self.norm(image_B),
            'mask_A': np.float32(mask_A),
            'mask_B': np.float32(mask_B),
            'consis_mask': np.float32(consis_mask),
            'P_A': np.float32(P_A),
            'P_B': np.float32(P_B),
            'consis_mask_idt_A': np.float32(consis_mask_idt_A),
            'consis_mask_idt_B': np.float32(consis_mask_idt_B),
            'mask_A_aug': np.float32(mask_A_aug),
            'mask_B_aug': np.float32(mask_B_aug)
        }

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        if self.mode == 'train':
            num_A = getattr(self, 'num_of_' + self.cls_list[0] + '_data')
            num_B = getattr(self, 'num_of_' + self.cls_list[1] + '_data')
            return max(num_A, num_B)
        elif self.mode == "test":
            num_A = getattr(self, 'num_of_' + self.cls_list[0] + '_data')
            num_B = getattr(self, 'num_of_' + self.cls_list[1] + '_data')
            return num_A * num_B
        return max(self.A_size, self.B_size)
