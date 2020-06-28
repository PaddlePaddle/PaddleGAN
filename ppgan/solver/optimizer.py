import copy
import paddle

from .lr_scheduler import build_lr_scheduler


def build_optimizer(cfg, parameter_list=None):
    cfg_copy = copy.deepcopy(cfg)
    
    lr_scheduler_cfg = cfg_copy.pop('lr_scheduler', None)

    lr_scheduler = build_lr_scheduler(lr_scheduler_cfg)

    opt_name = cfg_copy.pop('name')

    return getattr(paddle.optimizer, opt_name)(lr_scheduler, parameter_list=parameter_list, **cfg_copy)
