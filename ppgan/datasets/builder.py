import paddle
import numbers
import numpy as np
from paddle.imperative import ParallelEnv

from paddle.incubate.hapi.distributed import DistributedBatchSampler
from ..utils.registry import Registry


DATASETS = Registry("DATASETS")


def build_dataloader(cfg, is_train=True):
    dataset = DATASETS.get(cfg.name)(cfg)
    
    batch_size = cfg.get('batch_size', 1)

    # dataloader = DictDataLoader(dataset, batch_size, is_train)

    place = paddle.fluid.CUDAPlace(ParallelEnv().dev_id) \
                    if ParallelEnv().nranks > 1 else paddle.fluid.CUDAPlace(0)

    sampler = DistributedBatchSampler(
                dataset,
                batch_size=batch_size,
                shuffle=True if is_train else False,
                drop_last=True if is_train else False)

    dataloader = paddle.io.DataLoader(dataset,
                                      batch_sampler=sampler,
                                      places=place,
                                      num_workers=0)

    return dataloader