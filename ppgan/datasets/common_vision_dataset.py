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
from .transforms.builder import build_transforms


@DATASETS.register()
class CommonVisionDataset(BaseDataset):
    """
    Dataset for using paddle vision default datasets
    """
    def __init__(self, cfg):
        """Initialize this dataset class.

        Args:
            cfg (dict) -- stores all the experiment flags
        """
        super(CommonVisionDataset, self).__init__(cfg)

        dataset_cls = getattr(paddle.vision.datasets, cfg.pop('class_name'))
        transform = build_transforms(cfg.pop('transforms', None))
        self.return_cls = cfg.pop('return_cls', True)

        param_dict = {}
        param_names = list(dataset_cls.__init__.__code__.co_varnames)
        if 'transform' in param_names:
            param_dict['transform'] = transform
        for name in param_names:
            if name in cfg:
                param_dict[name] = cfg.get(name)

        self.dataset = dataset_cls(**param_dict)

    def __getitem__(self, index):
        return_dict = {}
        return_list = self.dataset[index]
        if isinstance(return_list, (tuple, list)):
            if len(return_list) == 2:
                return_dict['img'] = return_list[0]
                if self.return_cls:
                    return_dict['class_id'] = np.asarray(return_list[1])
            else:
                return_dict['img'] = return_list[0]
        else:
            return_dict['img'] = return_list

        return return_dict

    def __len__(self):
        return len(self.dataset)
