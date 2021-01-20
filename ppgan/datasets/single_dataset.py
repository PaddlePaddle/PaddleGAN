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
    def __init__(self, dataroot, preprocess):
        """Initialize single dataset class.

        Args:
            dataroot (str): Directory of dataset.
            preprocess (list[dict]): A sequence of data preprocess config.
        """
        super(SingleDataset, self).__init__(preprocess)
        self.dataroot = dataroot
        self.data_infos = self.prepare_data_infos()

    def prepare_data_infos(self):
        """prepare image paths from a folder.

        Returns:
            list[dict]: List that contains paired image paths.
        """
        data_infos = []
        paths = sorted(self.scan_folder(self.dataroot))
        for path in paths:
            data_infos.append(dict(A_path=path))

        return data_infos
