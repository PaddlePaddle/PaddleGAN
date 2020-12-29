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

import random
import os.path

from .base_dataset import BaseDataset
from .builder import DATASETS


@DATASETS.register()
class UnpairedDataset(BaseDataset):
    """
    """
    def __init__(self, dataroot_a, dataroot_b, max_size, is_train, preprocess):
        """Initialize unpaired dataset class.

        Args:
            dataroot_a (str): Directory of dataset a.
            dataroot_b (str): Directory of dataset b.
            max_size (int): max size of dataset size.
            is_train (int): whether in train mode.
            preprocess (list[dict]): A sequence of data preprocess config.

        """
        super(UnpairedDataset, self).__init__(preprocess)
        self.dir_A = os.path.join(dataroot_a)
        self.dir_B = os.path.join(dataroot_b)
        self.is_train = is_train
        self.data_infos_a = self.prepare_data_infos(self.dir_A)
        self.data_infos_b = self.prepare_data_infos(self.dir_B)
        self.size_a = len(self.data_infos_a)
        self.size_b = len(self.data_infos_b)

    def prepare_data_infos(self, dataroot):
        """Load unpaired image paths of one domain.

        Args:
            dataroot (str): Path to the folder root for unpaired images of
                one domain.

        Returns:
            list[dict]: List that contains unpaired image paths of one domain.
        """
        data_infos = []
        paths = sorted(self.scan_folder(dataroot))
        for path in paths:
            data_infos.append(dict(path=path))
        return data_infos

    def __getitem__(self, idx):
        if self.is_train:
            img_a_path = self.data_infos_a[idx % self.size_a]['path']
            idx_b = random.randint(0, self.size_b - 1)
            img_b_path = self.data_infos_b[idx_b]['path']
            datas = dict(A_path=img_a_path, B_path=img_b_path)
        else:
            img_a_path = self.data_infos_a[idx % self.size_a]['path']
            img_b_path = self.data_infos_b[idx % self.size_b]['path']
            datas = dict(A_path=img_a_path, B_path=img_b_path)

        if self.preprocess:
            datas = self.preprocess(datas)

        return datas

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.size_a, self.size_b)
