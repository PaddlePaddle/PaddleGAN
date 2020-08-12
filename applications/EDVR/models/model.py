#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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

import os
import wget
import logging
try:
    from configparser import ConfigParser
except:
    from ConfigParser import ConfigParser

import paddle.fluid as fluid
from .utils import download, AttrDict

WEIGHT_DIR = os.path.join(os.path.expanduser('~'), '.paddle', 'weights')

logger = logging.getLogger(__name__)


def is_parameter(var):
    return isinstance(var, fluid.framework.Parameter)


class NotImplementError(Exception):
    "Error: model function not implement"

    def __init__(self, model, function):
        super(NotImplementError, self).__init__()
        self.model = model.__class__.__name__
        self.function = function.__name__

    def __str__(self):
        return "Function {}() is not implemented in model {}".format(
            self.function, self.model)


class ModelNotFoundError(Exception):
    "Error: model not found"

    def __init__(self, model_name, avail_models):
        super(ModelNotFoundError, self).__init__()
        self.model_name = model_name
        self.avail_models = avail_models

    def __str__(self):
        msg = "Model {} Not Found.\nAvailiable models:\n".format(
            self.model_name)
        for model in self.avail_models:
            msg += "  {}\n".format(model)
        return msg


class ModelBase(object):
    def __init__(self, name, cfg, mode='train'):
        assert mode in ['train', 'valid', 'test', 'infer'], \
                "Unknown mode type {}".format(mode)
        self.name = name
        self.is_training = (mode == 'train')
        self.mode = mode
        self.cfg = cfg
        self.dataloader = None

    def build_model(self):
        "build model struct"
        raise NotImplementError(self, self.build_model)

    def build_input(self, use_dataloader):
        "build input Variable"
        raise NotImplementError(self, self.build_input)

    def optimizer(self):
        "get model optimizer"
        raise NotImplementError(self, self.optimizer)

    def outputs():
        "get output variable"
        raise notimplementerror(self, self.outputs)

    def loss(self):
        "get loss variable"
        raise notimplementerror(self, self.loss)

    def feeds(self):
        "get feed inputs list"
        raise NotImplementError(self, self.feeds)

    def fetches(self):
        "get fetch list of model"
        raise NotImplementError(self, self.fetches)

    def weights_info(self):
        "get model weight default path and download url"
        raise NotImplementError(self, self.weights_info)

    def get_weights(self):
        "get model weight file path, download weight from Paddle if not exist"
        path, url = self.weights_info()
        path = os.path.join(WEIGHT_DIR, path)
        if not os.path.isdir(WEIGHT_DIR):
            logger.info('{} not exists, will be created automatically.'.format(
                WEIGHT_DIR))
            os.makedirs(WEIGHT_DIR)
        if os.path.exists(path):
            return path

        logger.info("Download weights of {} from {}".format(self.name, url))
        wget.download(url, path)
        return path

    def dataloader(self):
        return self.dataloader

    def epoch_num(self):
        "get train epoch num"
        return self.cfg.TRAIN.epoch

    def pretrain_info(self):
        "get pretrain base model directory"
        return (None, None)

    def get_pretrain_weights(self):
        "get model weight file path, download weight from Paddle if not exist"
        path, url = self.pretrain_info()
        if not path:
            return None

        path = os.path.join(WEIGHT_DIR, path)
        if not os.path.isdir(WEIGHT_DIR):
            logger.info('{} not exists, will be created automatically.'.format(
                WEIGHT_DIR))
            os.makedirs(WEIGHT_DIR)
        if os.path.exists(path):
            return path

        logger.info("Download pretrain weights of {} from {}".format(self.name,
                                                                     url))
        download(url, path)
        return path

    def load_pretrain_params(self, exe, pretrain, prog, place):
        logger.info("Load pretrain weights from {}".format(pretrain))
        state_dict = fluid.load_program_state(pretrain)
        fluid.set_program_state(prog, state_dict)

    def load_test_weights(self, exe, weights, prog, place):
        params_list = list(filter(is_parameter, prog.list_vars()))
        fluid.load(prog, weights, executor=exe, var_list=params_list)

    def get_config_from_sec(self, sec, item, default=None):
        if sec.upper() not in self.cfg:
            return default
        return self.cfg[sec.upper()].get(item, default)


class ModelZoo(object):
    def __init__(self):
        self.model_zoo = {}

    def regist(self, name, model):
        assert model.__base__ == ModelBase, "Unknow model type {}".format(
            type(model))
        self.model_zoo[name] = model

    def get(self, name, cfg, mode='train'):
        for k, v in self.model_zoo.items():
            if k.upper() == name.upper():
                return v(name, cfg, mode)
        raise ModelNotFoundError(name, self.model_zoo.keys())


# singleton model_zoo
model_zoo = ModelZoo()


def regist_model(name, model):
    model_zoo.regist(name, model)


def get_model(name, cfg, mode='train'):
    return model_zoo.get(name, cfg, mode)
