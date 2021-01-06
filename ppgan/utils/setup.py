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

import os
import time
import paddle

from .logger import setup_logger


def setup(args, cfg):
    if args.evaluate_only:
        cfg.is_train = False
    else:
        cfg.is_train = True

    cfg.timestamp = time.strftime('-%Y-%m-%d-%H-%M', time.localtime())
    cfg.output_dir = os.path.join(
        cfg.output_dir,
        os.path.splitext(os.path.basename(str(args.config_file)))[0] +
        cfg.timestamp)

    logger = setup_logger(cfg.output_dir)

    logger.info('Configs: {}'.format(cfg))

    if paddle.is_compiled_with_cuda():
        paddle.set_device('gpu')
    else:
        paddle.set_device('cpu')
