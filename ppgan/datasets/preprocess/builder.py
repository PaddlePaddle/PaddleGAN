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
import paddle
from ...utils.registry import Registry, build_from_config

LOAD_PIPELINE = Registry("LOAD_PIPELINE")
TRANSFORMS = Registry("TRANSFORM")


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
    def __init__(self, functions, input_keys=None):
        self.functions = functions
        self.input_keys = input_keys

    def __call__(self, datas):
        if self.input_keys:
            data = [datas[key] for key in self.input_keys]
            data = tuple(data)
        else:
            data = datas

        for func in self.functions:
            try:
                data = func(data)
            except Exception as e:
                stack_info = traceback.format_exc()
                print("fail to perform fuction [{}] with error: "
                      "{} and stack:\n{}".format(func, e, str(stack_info)))
                raise RuntimeError

        if self.input_keys:
            for i, key in enumerate(self.input_keys):
                datas[key] = data[i]
        else:
            datas = data
        return datas


def build_load_pipeline(cfg):
    load_pipeline = []
    if not isinstance(cfg, (list, tuple)):
        cfg = [cfg]

    for cfg_ in cfg:
        load_func = build_from_config(cfg_, LOAD_PIPELINE)
        load_pipeline.append(load_func)

    load_pipeline = Compose(load_pipeline)
    return load_pipeline


def build_transforms(cfg):
    transforms = []
    # print('debug:', cfg)
    # input_keys = cfg.pop('input_keys', None)
    # if not isinstance(cfg, (list, tuple)):
    #     cfg = [cfg]
    cfg_ = cfg.copy()
    input_keys = None
    if 'input_keys' in cfg_:
        input_keys = cfg_.pop('input_keys')
        trans_cfg = cfg_['pipeline']
    else:
        trans_cfg = cfg_

    for trans_cfg_ in trans_cfg:
        transform = build_from_config(trans_cfg_, TRANSFORMS)
        transforms.append(transform)

    transforms = Compose(transforms, input_keys=input_keys)
    return transforms


# def build_preprocess(cfgs):
#     preprocess = []

#     for cfg in cfgs:
#         cfg_ = copy.deepcopy(cfg)
#         name = cfg_.pop('name')
#         preprocess.append(PREPROCESS.get(name)(**cfg_))

#     preprocess = Compose(preprocess)
#     return preprocess
