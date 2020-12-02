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

from .builder import DATASETS
from .base_dataset import BaseDataset


@DATASETS.register()
class PairedDataset(BaseDataset):
    """A dataset class for paired image dataset.
    """
    def __init__(self, dataroot, preprocess):
        """Initialize this dataset class.

        Args:
            dataroot (str): Directory of dataset.
            preprocess (list[dict]): A sequence of data preprocess config.

        """
        super(PairedDataset, self).__init__(preprocess)
        self.dataroot = dataroot
        self.data_infos = self.prepare_data_infos()

    def prepare_data_infos(self):
        """Load paired image paths.

        Returns:
            list[dict]: List that contains paired image paths.
        """
        data_infos = []
        pair_paths = sorted(self.scan_folder(self.dataroot))
        for pair_path in pair_paths:
            data_infos.append(dict(pair_path=pair_path))

        return data_infos
