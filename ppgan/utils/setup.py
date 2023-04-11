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
import yaml
import paddle
import numpy as np
import random
from .config import cfg2dict
from .logger import setup_logger


def setup(args, cfg):
    if args.evaluate_only:
        cfg.is_train = False
    else:
        cfg.is_train = True

    if args.profiler_options:
        cfg.profiler_options = args.profiler_options
    else:
        cfg.profiler_options = None

    cfg.timestamp = time.strftime('-%Y-%m-%d-%H-%M', time.localtime())
    cfg.output_dir = os.path.join(
        cfg.output_dir,
        os.path.splitext(os.path.basename(str(args.config_file)))[0] +
        cfg.timestamp)

    logger = setup_logger(cfg.output_dir)

    logger.info('Configs: \n{}'.format(yaml.dump(cfg2dict(cfg))))

    if paddle.is_compiled_with_cuda():
        paddle.set_device('gpu')
    # paddle.is_compiled_with_npu() will be aborted after paddle-2.4
    elif int(paddle.version.major) != 0 and int(
            paddle.version.major) <= 2 and int(
                paddle.version.minor) <= 4 and paddle.is_compiled_with_npu():
        paddle.set_device('npu')
    elif paddle.is_compiled_with_custom_device("npu"):
        paddle.set_device('npu')
    elif paddle.is_compiled_with_xpu():
        paddle.set_device('xpu')
    else:
        paddle.set_device('cpu')

    if args.seed:
        paddle.seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        paddle.framework.random._manual_program_seed(args.seed)

    # add amp and amp_level args into cfg
    cfg['amp'] = args.amp
    cfg['amp_level'] = args.amp_level
