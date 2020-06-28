import copy
from ...utils.registry import Registry


DISCRIMINATORS = Registry("DISCRIMINATOR")


def build_discriminator(cfg):
    cfg_copy = copy.deepcopy(cfg)
    name = cfg_copy.pop('name')
    discriminator = DISCRIMINATORS.get(name)(**cfg_copy)
    return discriminator
