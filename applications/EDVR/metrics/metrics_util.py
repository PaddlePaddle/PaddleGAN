#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import logging

import numpy as np
import json
from metrics.edvr_metrics import edvr_metrics as edvr_metrics

logger = logging.getLogger(__name__)


class Metrics(object):
    def __init__(self, name, mode, metrics_args):
        """Not implemented"""
        pass

    def calculate_and_log_out(self, fetch_list, info=''):
        """Not implemented"""
        pass

    def accumulate(self, fetch_list, info=''):
        """Not implemented"""
        pass

    def finalize_and_log_out(self, info='', savedir='./'):
        """Not implemented"""
        pass

    def reset(self):
        """Not implemented"""
        pass


class EDVRMetrics(Metrics):
    def __init__(self, name, mode, cfg):
        self.name = name
        self.mode = mode
        args = {}
        args['mode'] = mode
        args['name'] = name
        self.calculator = edvr_metrics.MetricsCalculator(**args)

    def calculate_and_log_out(self, fetch_list, info=''):
        if (self.mode == 'train') or (self.mode == 'valid'):
            loss = np.array(fetch_list[0])
            logger.info(info + '\tLoss = {}'.format('%.04f' % np.mean(loss)))
        elif self.mode == 'test':
            pass

    def accumulate(self, fetch_list):
        self.calculator.accumulate(fetch_list)

    def finalize_and_log_out(self, info='', savedir='./'):
        self.calculator.finalize_metrics(savedir)

    def reset(self):
        self.calculator.reset()


class MetricsZoo(object):
    def __init__(self):
        self.metrics_zoo = {}

    def regist(self, name, metrics):
        assert metrics.__base__ == Metrics, "Unknow model type {}".format(
            type(metrics))
        self.metrics_zoo[name] = metrics

    def get(self, name, mode, cfg):
        for k, v in self.metrics_zoo.items():
            if k == name:
                return v(name, mode, cfg)
        raise MetricsNotFoundError(name, self.metrics_zoo.keys())


# singleton metrics_zoo
metrics_zoo = MetricsZoo()


def regist_metrics(name, metrics):
    metrics_zoo.regist(name, metrics)


def get_metrics(name, mode, cfg):
    return metrics_zoo.get(name, mode, cfg)


# sort by alphabet
regist_metrics("EDVR", EDVRMetrics)
