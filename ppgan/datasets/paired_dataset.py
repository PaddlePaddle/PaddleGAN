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
import os.path
from .base_dataset import BaseDataset, get_params, get_transform
from .image_folder import make_dataset

from .builder import DATASETS
from .transforms.builder import build_transforms


@DATASETS.register()
class PairedDataset(BaseDataset):
    """A dataset class for paired image dataset.
    """
    def __init__(self, cfg):
        """Initialize this dataset class.

        Args:
            cfg (dict): configs of datasets.
        """
        BaseDataset.__init__(self, cfg)
        self.dir_AB = os.path.join(cfg.dataroot,
                                   cfg.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(
            self.dir_AB, cfg.max_dataset_size))  # get image paths

        self.input_nc = self.cfg.output_nc if self.cfg.direction == 'BtoA' else self.cfg.input_nc
        self.output_nc = self.cfg.input_nc if self.cfg.direction == 'BtoA' else self.cfg.output_nc
        self.transforms = build_transforms(cfg.transforms)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        AB = cv2.cvtColor(cv2.imread(AB_path), cv2.COLOR_BGR2RGB)

        # split AB image into A and B
        h, w = AB.shape[:2]
        # w, h = AB.size
        w2 = int(w / 2)

        A = AB[:h, :w2, :]
        B = AB[:h, w2:, :]

        # apply the same transform to both A and B
        A, B = self.transforms((A, B))

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
