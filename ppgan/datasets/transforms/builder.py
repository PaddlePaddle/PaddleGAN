import copy
import traceback
import paddle
from ...utils.registry import Registry

TRANSFORMS = Registry("TRANSFORMS")


class Compose(object):
    """
    Composes several transforms together use for composing list of transforms
    together for a dataset transform.

    Args:
        transforms (list): List of transforms to compose.

    Returns:
        A compose object which is callable, __call__ for this Compose
        object will call each given :attr:`transforms` sequencely.

    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for f in self.transforms:
            try:
                data = f(data)
            except Exception as e:
                print(f)
                stack_info = traceback.format_exc()
                print("fail to perform transform [{}] with error: "
                      "{} and stack:\n{}".format(f, e, str(stack_info)))
                raise e
        return data


def build_transforms(cfg):
    transforms = []

    for trans_cfg in cfg:
        temp_trans_cfg = copy.deepcopy(trans_cfg)
        name = temp_trans_cfg.pop('name')
        transforms.append(TRANSFORMS.get(name)(**temp_trans_cfg))

    transforms = Compose(transforms)
    return transforms
