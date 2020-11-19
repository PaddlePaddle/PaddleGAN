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
