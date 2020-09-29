import paddle

from ..utils.registry import Registry

MODELS = Registry("MODEL")


def build_model(cfg):
    model = MODELS.get(cfg.model.name)(cfg)
    return model
