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

import copy
import traceback

from ...utils.registry import Registry, build_from_config

LOAD_PIPELINE = Registry("LOAD_PIPELINE")
TRANSFORMS = Registry("TRANSFORM")
PREPROCESS = Registry("PREPROCESS")


class Compose(object):
    """
    Composes several transforms together use for composing list of transforms
    together for a dataset transform.

    Args:
        functions (list[callable]): List of functions to compose.

    Returns:
        A compose object which is callable, __call__ for this Compose
        object will call each given :attr:`transforms` sequencely.

    """
    def __init__(self, functions):
        self.functions = functions

    def __call__(self, datas):

        for func in self.functions:
            try:
                datas = func(datas)
            except Exception as e:
                stack_info = traceback.format_exc()
                print("fail to perform fuction [{}] with error: "
                      "{} and stack:\n{}".format(func, e, str(stack_info)))
                raise RuntimeError
        return datas


def build_preprocess(cfg):
    preproccess = []
    if not isinstance(cfg, (list, tuple)):
        cfg = [cfg]

    for cfg_ in cfg:
        process = build_from_config(cfg_, PREPROCESS)
        preproccess.append(process)

    preproccess = Compose(preproccess)
    return preproccess


def build_transforms(cfg):
    transforms = []

    for trans_cfg in cfg:
        temp_trans_cfg = copy.deepcopy(trans_cfg)
        name = temp_trans_cfg.pop('name')
        transforms.append(TRANSFORMS.get(name)(**temp_trans_cfg))

    transforms = Compose(transforms)
    return transforms
