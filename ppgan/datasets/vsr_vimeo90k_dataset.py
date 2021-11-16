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

import os
import cv2
import glob
import random
import logging
import numpy as np
from paddle.io import Dataset

from .base_sr_dataset import BaseDataset
from .builder import DATASETS


@DATASETS.register()
class VSRVimeo90KDataset(BaseDataset):
    """Vimeo90K dataset for video super resolution for recurrent networks.

    The dataset loads several LQ (Low-Quality) frames and GT (Ground-Truth)
    frames. Then it applies specified transforms and finally returns a dict
    containing paired data and other information.

    It reads Vimeo90K keys from the txt file. Each line contains video frame folder

    Examples:

        00001/0233
        00001/0234

    Args:
        lq_folder (str): Path to a low quality image folder.
        gt_folder (str): Path to a ground truth image folder.
        ann_file (str): Path to the annotation file.
        preprocess (list[dict|callable]): A list functions of data transformations.
    """
    def __init__(self, lq_folder, gt_folder, ann_file, preprocess):
        super().__init__(preprocess)

        self.lq_folder = str(lq_folder)
        self.gt_folder = str(gt_folder)
        self.ann_file = str(ann_file)

        self.data_infos = self.prepare_data_infos()

    def prepare_data_infos(self):

        with open(self.ann_file, 'r') as fin:
            keys = [line.strip() for line in fin]

        data_infos = []
        for key in keys:
            lq_paths = sorted(
                glob.glob(os.path.join(self.lq_folder, key, '*.png')))
            gt_paths = sorted(
                glob.glob(os.path.join(self.gt_folder, key, '*.png')))

            data_infos.append(dict(lq_path=lq_paths, gt_path=gt_paths, key=key))

        return data_infos
