import copy
from ...utils.registry import Registry


GENERATORS = Registry("GENERATOR")


def build_generator(cfg):
    cfg_copy = copy.deepcopy(cfg)
    name = cfg_copy.pop('name')
    generator = GENERATORS.get(name)(**cfg_copy)
    return generator
