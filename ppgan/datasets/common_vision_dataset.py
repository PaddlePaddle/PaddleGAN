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

import numpy as np
import paddle

from .builder import DATASETS
from .base_dataset import BaseDataset
from .preprocess.builder import build_transforms


@DATASETS.register()
class CommonVisionDataset(paddle.io.Dataset):
    """
    Dataset for using paddle vision default datasets, such as mnist, flowers.
    """
    def __init__(self,
                 dataset_name,
                 transforms=None,
                 return_label=True,
                 params=None):
        """Initialize this dataset class.

        Args:
            dataset_name (str): return a dataset from paddle.vision.datasets by this option.
            transforms (list[dict]): A sequence of data transforms config.
            return_label (bool): whether to retuan a label of a sample.
            params (dict): paramters of paddle.vision.datasets.
        """
        super(CommonVisionDataset, self).__init__()

        dataset_cls = getattr(paddle.vision.datasets, dataset_name)
        transform = build_transforms(transforms)
        self.return_label = return_label

        param_dict = {}
        param_names = list(dataset_cls.__init__.__code__.co_varnames)
        if 'transform' in param_names:
            param_dict['transform'] = transform

        if params is not None:
            for name in param_names:
                if name in params:
                    param_dict[name] = params[name]

        self.dataset = dataset_cls(**param_dict)

    def __getitem__(self, index):
        return_dict = {}
        return_list = self.dataset[index]
        if isinstance(return_list, (tuple, list)):
            if len(return_list) == 2:
                return_dict['img'] = return_list[0]
                if self.return_label:
                    return_dict['class_id'] = np.asarray(return_list[1])
            else:
                return_dict['img'] = return_list[0]
        else:
            return_dict['img'] = return_list

        return return_dict

    def __len__(self):
        return len(self.dataset)
