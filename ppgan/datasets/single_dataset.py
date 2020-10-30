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

import cv2
import paddle
from .base_dataset import BaseDataset, get_transform
from .image_folder import make_dataset

from .builder import DATASETS
from .transforms.builder import build_transforms


@DATASETS.register()
class SingleDataset(BaseDataset):
    """
    """
    def __init__(self, cfg):
        """Initialize this dataset class.

        Args:
            cfg (dict) -- stores all the experiment flags
        """
        BaseDataset.__init__(self, cfg)
        self.A_paths = sorted(make_dataset(cfg.dataroot, cfg.max_dataset_size))
        input_nc = self.cfg.output_nc if self.cfg.direction == 'BtoA' else self.cfg.input_nc
        self.transform = build_transforms(self.cfg.transforms)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        A_path = self.A_paths[index]
        A_img = cv2.cvtColor(cv2.imread(A_path), cv2.COLOR_BGR2RGB)
        A = self.transform(A_img)

        return {'A': A, 'A_paths': A_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)

    def get_path_by_indexs(self, indexs):
        if isinstance(indexs, paddle.Variable):
            indexs = indexs.numpy()
        current_paths = []
        for index in indexs:
            current_paths.append(self.A_paths[index])
        return current_paths
