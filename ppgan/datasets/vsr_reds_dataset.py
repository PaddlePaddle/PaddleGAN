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

from .builder import DATASETS
from .base_sr_dataset import BaseDataset

logger = logging.getLogger(__name__)


@DATASETS.register()
class VSRREDSDataset(BaseDataset):
    """REDS dataset for video super resolution for Sliding-window networks.

    The dataset loads several LQ (Low-Quality) frames and a center GT
    (Ground-Truth) frame. Then it applies specified transforms and finally
    returns a dict containing paired data and other information.

    It reads REDS keys from the txt file. Each line contains video frame folder

    Examples:

        000/00000000.png (720, 1280, 3)
        000/00000001.png (720, 1280, 3)

    Args:
        lq_folder (str): Path to a low quality image folder.
        gt_folder (str): Path to a ground truth image folder.
        ann_file (str): Path to the annotation file.
        num_frames (int): Window size for input frames.
        preprocess (list[dict|callable]): A list functions of data transformations.
        val_partition (str): Validation partition mode. Choices ['official' or 'REDS4']. Default: 'REDS4'.
        test_mode (bool): Store `True` when building test dataset. Default: `False`.
    """
    def __init__(self,
                 lq_folder,
                 gt_folder,
                 ann_file,
                 num_frames,
                 preprocess,
                 val_partition='REDS4',
                 test_mode=False):
        super().__init__(preprocess)
        assert num_frames % 2 == 1, (f'num_frames should be odd numbers, '
                                     f'but received {num_frames }.')
        self.lq_folder = str(lq_folder)
        self.gt_folder = str(gt_folder)
        self.ann_file = str(ann_file)
        self.num_frames = num_frames
        self.val_partition = val_partition
        self.test_mode = test_mode
        self.data_infos = self.prepare_data_infos()

    def prepare_data_infos(self):
        """Load annoations for REDS dataset.
        Returns:
            dict: Returned dict for LQ and GT pairs.
        """
        # get keys
        with open(self.ann_file, 'r') as fin:
            keys = [v.strip().split('.')[0] for v in fin]

        if self.val_partition == 'REDS4':
            val_partition = ['000', '011', '015', '020']
        elif self.val_partition == 'official':
            val_partition = [f'{v:03d}' for v in range(240, 270)]
        else:
            raise ValueError(f'Wrong validation partition {self.val_partition}.'
                             f'Supported ones are ["official", "REDS4"]')

        if self.test_mode:
            keys = [v for v in keys if v.split('/')[0] in val_partition]
        else:
            keys = [v for v in keys if v.split('/')[0] not in val_partition]

        data_infos = []
        for key in keys:
            data_infos.append(
                dict(lq_path=self.lq_folder,
                     gt_path=self.gt_folder,
                     key=key,
                     max_frame_num=100,
                     num_frames=self.num_frames))

        return data_infos
