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

from .base_dataset import BaseDataset
from .builder import DATASETS


@DATASETS.register()
class SingleDataset(BaseDataset):
    """
    """
    def __init__(self, dataroot, load_pipeline, transforms):
        """Initialize single dataset class.

        Args:
            dataroot (str): Directory of dataset.
            load_pipeline (list[dict]): A sequence of data loading config.
            transforms (list[dict]): A sequence of data transform config.
        """
        BaseDataset.__init__(self, load_pipeline, transforms)
        self.dataroot = dataroot
        self.annotations = self.load_annotations()

    def load_annotations(self):
        """Load paired image paths.

        Returns:
            list[dict]: List that contains paired image paths.
        """
        annotations = []
        paths = sorted(self.scan_folder(self.dataroot))
        for path in paths:
            annotations.append(dict(A_path=path))

        return annotations
