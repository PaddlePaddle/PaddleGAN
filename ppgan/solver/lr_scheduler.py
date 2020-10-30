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

import paddle


def build_lr_scheduler(cfg):
    name = cfg.pop('name')

    # TODO: add more learning rate scheduler
    if name == 'linear':

        def lambda_rule(epoch):
            lr_l = 1.0 - max(
                0, epoch + 1 - cfg.start_epoch) / float(cfg.decay_epochs + 1)
            return lr_l

        scheduler = paddle.optimizer.lr.LambdaDecay(cfg.learning_rate,
                                                    lr_lambda=lambda_rule)
        return scheduler
    else:
        raise NotImplementedError
