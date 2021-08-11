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

import time
import paddle
import numbers
import numpy as np

from paddle.distributed import ParallelEnv
from paddle.io import DistributedBatchSampler

from .repeat_dataset import RepeatDataset
from ..utils.registry import Registry, build_from_config

DATASETS = Registry("DATASETS")


def build_dataset(cfg):
    name = cfg.pop('name')

    if name == 'RepeatDataset':
        dataset_ = build_from_config(cfg['dataset'], DATASETS)
        dataset = RepeatDataset(dataset_, cfg['times'])
    else:
        dataset = dataset = DATASETS.get(name)(**cfg)

    return dataset


def build_dataloader(cfg, is_train=True, distributed=True):
    cfg_ = cfg.copy()

    batch_size = cfg_.pop('batch_size', 1)
    num_workers = cfg_.pop('num_workers', 0)
    use_shared_memory = cfg_.pop('use_shared_memory', True)

    dataset = build_dataset(cfg_)

    if distributed:
        sampler = DistributedBatchSampler(dataset,
                                          batch_size=batch_size,
                                          shuffle=True if is_train else False,
                                          drop_last=True if is_train else False)

        dataloader = paddle.io.DataLoader(dataset,
                                          batch_sampler=sampler,
                                          num_workers=num_workers,
                                          use_shared_memory=use_shared_memory)
    else:
        dataloader = paddle.io.DataLoader(dataset,
                                          batch_size=batch_size,
                                          shuffle=True if is_train else False,
                                          drop_last=True if is_train else False,
                                          use_shared_memory=use_shared_memory,
                                          num_workers=num_workers)

    return dataloader
