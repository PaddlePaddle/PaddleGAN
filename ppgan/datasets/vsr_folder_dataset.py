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

logger = logging.getLogger(__name__)


@DATASETS.register()
class VSRFolderDataset(BaseDataset):
    """Video super-resolution for folder format.

    Args:
        lq_folder (str): Path to a low quality image folder.
        gt_folder (str): Path to a ground truth image folder.
        ann_file (str): Path to the annotation file.
        preprocess (list[dict|callable]): A list functions of data transformations.
        num_frames (int): Number of frames of each input clip.
        times (int): Repeat times of datset length.
    """
    def __init__(self,
                 lq_folder,
                 gt_folder,
                 preprocess,
                 num_frames=None,
                 times=1):
        super().__init__(preprocess)

        self.lq_folder = str(lq_folder)
        self.gt_folder = str(gt_folder)
        self.num_frames = num_frames
        self.times = times

        self.data_infos = self.prepare_data_infos()

    def prepare_data_infos(self):

        sequences = sorted(glob.glob(os.path.join(self.lq_folder, '*')))

        data_infos = []
        for sequence in sequences:
            sequence_length = len(glob.glob(os.path.join(sequence, '*.png')))
            if self.num_frames is None:
                num_frames = sequence_length
            else:
                num_frames = self.num_frames
            data_infos.append(
                dict(lq_path=self.lq_folder,
                     gt_path=self.gt_folder,
                     key=sequence.replace(f'{self.lq_folder}/', ''),
                     num_frames=num_frames,
                     sequence_length=sequence_length))
        return data_infos
