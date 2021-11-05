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

import inspect
import traceback


class Registry(object):
    """
    The registry that provides name -> object mapping, to support third-party users' custom modules.
    To create a registry (inside ppgan):
    .. code-block:: python
        BACKBONE_REGISTRY = Registry('BACKBONE')
    To register an object:
    .. code-block:: python
        @BACKBONE_REGISTRY.register()
        class MyBackbone():
            ...
    Or:
    .. code-block:: python
        BACKBONE_REGISTRY.register(MyBackbone)
    """
    def __init__(self, name):
        """
        Args:
            name (str): the name of this registry
        """
        self._name = name

        self._obj_map = {}

    def _do_register(self, name, obj):
        assert (
            name not in self._obj_map
        ), "An object named '{}' was already registered in '{}' registry!".format(
            name, self._name)
        self._obj_map[name] = obj

    def register(self, obj=None, name=None):
        """
        Register the given object under the the name `obj.__name__`.
        Can be used as either a decorator or not. See docstring of this class for usage.
        """
        if obj is None:
            # used as a decorator
            def deco(func_or_class, name=name):
                if name is None:
                    name = func_or_class.__name__
                self._do_register(name, func_or_class)
                return func_or_class

            return deco

        # used as a function call
        if name is None:
            name = obj.__name__
        self._do_register(name, obj)

    def get(self, name):
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError(
                "No object named '{}' found in '{}' registry!".format(
                    name, self._name))

        return ret


# code was based on mmcv
# Copyright (c) Copyright (c) OpenMMLab.
def build_from_config(cfg, registry, default_args=None):
    """Build a class from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "name".
        registry (ppgan.utils.Registry): The registry to search the name from.
        default_args (dict, optional): Default initialization arguments.

    Returns:
        class: The constructed class.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
    if 'name' not in cfg:
        if default_args is None or 'name' not in default_args:
            raise KeyError(
                '`cfg` or `default_args` must contain the key "name", '
                f'but got {cfg}\n{default_args}')
    if not isinstance(registry, Registry):
        raise TypeError('registry must be an ppgan.utils.Registry object, '
                        f'but got {type(registry)}')
    if not (isinstance(default_args, dict) or default_args is None):
        raise TypeError('default_args must be a dict or None, '
                        f'but got {type(default_args)}')

    args = cfg.copy()

    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)

    cls_name = args.pop('name')
    if isinstance(cls_name, str):
        obj_cls = registry.get(cls_name)
    elif inspect.isclass(cls_name):
        obj_cls = obj_cls
    else:
        raise TypeError(
            f'name must be a str or valid name, but got {type(cls_name)}')

    try:
        instance = obj_cls(**args)
    except Exception as e:
        stack_info = traceback.format_exc()
        print("Fail to initial class [{}] with error: "
              "{} and stack:\n{}".format(cls_name, e, str(stack_info)))
        raise e
    return instance
