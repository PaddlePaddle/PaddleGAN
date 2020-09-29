import os
import time
import paddle

from paddle.distributed import ParallelEnv

from .logger import setup_logger


def setup(args, cfg):
    if args.evaluate_only:
        cfg.isTrain = False

    cfg.timestamp = time.strftime('-%Y-%m-%d-%H-%M', time.localtime())
    cfg.output_dir = os.path.join(cfg.output_dir,
                                  str(cfg.model.name) + cfg.timestamp)

    logger = setup_logger(cfg.output_dir)

    logger.info('Configs: {}'.format(cfg))

    place = paddle.CUDAPlace(ParallelEnv().dev_id) \
                    if ParallelEnv().nranks > 1 else paddle.CUDAPlace(0)
    paddle.disable_static(place)
