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

import os
import copy

from pathlib import Path
from .base_dataset import BaseDataset
from .builder import DATASETS


@DATASETS.register()
class SRDataset(BaseDataset):
    """Base super resulotion dataset for image restoration."""
    def __init__(self,
                 lq_folder,
                 gt_folder,
                 preprocess,
                 scale,
                 filename_tmpl='{}'):
        super(SRDataset, self).__init__(preprocess)
        self.lq_folder = lq_folder
        self.gt_folder = gt_folder
        self.scale = scale
        self.filename_tmpl = filename_tmpl

        self.prepare_data_infos()

    def prepare_data_infos(self):
        """Load annoations for SR dataset.

        It loads the LQ and GT image path from folders.

        Returns:
            dict: Returned dict for LQ and GT pairs.
        """
        self.data_infos = []
        lq_paths = self.scan_folder(self.lq_folder)
        gt_paths = self.scan_folder(self.gt_folder)
        assert len(lq_paths) == len(gt_paths), (
            f'gt and lq datasets have different number of images: '
            f'{len(lq_paths)}, {len(gt_paths)}.')
        for gt_path in gt_paths:
            basename, ext = os.path.splitext(os.path.basename(gt_path))
            lq_path = os.path.join(self.lq_folder,
                                   (f'{self.filename_tmpl.format(basename)}'
                                    f'{ext}'))
            assert lq_path in lq_paths, f'{lq_path} is not in lq_paths.'
            self.data_infos.append(dict(lq_path=lq_path, gt_path=gt_path))
        return self.data_infos
