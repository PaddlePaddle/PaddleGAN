import os
import time
import paddle

from .logger import setup_logger


def setup(args, cfg):
    if args.evaluate_only:
        cfg.is_train = False
    else:
        cfg.is_train = True

    cfg.timestamp = time.strftime('-%Y-%m-%d-%H-%M', time.localtime())
    cfg.output_dir = os.path.join(cfg.output_dir,
                                  str(cfg.model.name) + cfg.timestamp)

    logger = setup_logger(cfg.output_dir)

    logger.info('Configs: {}'.format(cfg))

    if paddle.is_compiled_with_cuda():
        paddle.set_device('gpu')
    else:
        paddle.set_device('cpu')
