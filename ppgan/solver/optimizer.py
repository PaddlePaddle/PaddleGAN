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
import paddle

from .lr_scheduler import build_lr_scheduler


def build_optimizer(cfg, lr_scheduler, parameter_list=None):
    cfg_copy = copy.deepcopy(cfg)

    opt_name = cfg_copy.pop('name')

    return getattr(paddle.optimizer, opt_name)(lr_scheduler,
                                               parameters=parameter_list,
                                               **cfg_copy)
