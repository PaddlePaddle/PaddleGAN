import copy
import paddle

from .lr_scheduler import build_lr_scheduler


def build_optimizer(cfg, lr_scheduler, parameter_list=None):
    cfg_copy = copy.deepcopy(cfg)

    opt_name = cfg_copy.pop('name')

    return getattr(paddle.optimizer, opt_name)(lr_scheduler,
                                               parameters=parameter_list,
                                               **cfg_copy)
