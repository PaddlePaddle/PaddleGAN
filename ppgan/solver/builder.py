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

from ..utils.registry import Registry

LRSCHEDULERS = Registry("LRSCHEDULER")
OPTIMIZERS = Registry("OPTIMIZER")


def build_lr_scheduler(cfg):
    cfg_ = cfg.copy()
    name = cfg_.pop('name')
    return LRSCHEDULERS.get(name)(**cfg_)


def build_optimizer(cfg, lr_scheduler, parameters=None):
    cfg_ = cfg.copy()
    name = cfg_.pop('name')
    return OPTIMIZERS.get(name)(lr_scheduler, parameters=parameters, **cfg_)
