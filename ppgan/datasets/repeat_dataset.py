#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

import paddle


class RepeatDataset(paddle.io.Dataset):
    """A wrapper of repeated dataset.

    The length of repeated dataset will be `times` larger than the original
    dataset. This is useful when the data loading time is long but the dataset
    is small. Using RepeatDataset can reduce the data loading time between
    epochs.

    Args:
        dataset (:obj:`Dataset`): The dataset to be repeated.
        times (int): Repeat times.
    """

    def __init__(self, dataset, times):
        self.dataset = dataset
        self.times = times

        self._ori_len = len(self.dataset)

    def __getitem__(self, idx):
        """Get item at each call.

        Args:
            idx (int): Index for getting each item.
        """
        return self.dataset[idx % self._ori_len]

    def __len__(self):
        """Length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return self.times * self._ori_len
