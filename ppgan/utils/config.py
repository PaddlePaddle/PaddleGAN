# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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
import yaml

from .logger import get_logger

__all__ = ['get_config']


class AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self[key] = value


def create_attr_dict(yaml_config):
    from ast import literal_eval
    for key, value in yaml_config.items():
        if type(value) is dict:
            yaml_config[key] = value = AttrDict(value)
        if isinstance(value, str):
            try:
                value = literal_eval(value)
            except BaseException:
                pass
        if isinstance(value, AttrDict):
            create_attr_dict(yaml_config[key])
        else:
            yaml_config[key] = value


def parse_config(cfg_file):
    """Load a config file into AttrDict"""
    with open(cfg_file, 'r') as fopen:
        yaml_config = AttrDict(yaml.load(fopen, Loader=yaml.SafeLoader))
    create_attr_dict(yaml_config)
    return yaml_config

Color = {
    'RED': '\033[31m',
    'HEADER': '\033[35m',  # deep purple
    'PURPLE': '\033[95m',  # purple
    'OKBLUE': '\033[94m',
    'OKGREEN': '\033[92m',
    'WARNING': '\033[93m',
    'FAIL': '\033[91m',
    'ENDC': '\033[0m'
}


def coloring(message, color="OKGREEN"):
    assert color in Color.keys()
    if os.environ.get('PADDLEGAN_COLORING', False):
        return Color[color] + str(message) + Color["ENDC"]
    else:
        return message

def print_dict(d, logger, delimiter=0):
    """
    Recursively visualize a dict and
    indenting acrrording by the relationship of keys.
    """
    placeholder = "-" * 60
    for k, v in sorted(d.items()):
        if isinstance(v, dict):
            logger.info("{}{} : ".format(delimiter * " ",
                                         coloring(k, "HEADER")))
            print_dict(v, logger, delimiter + 4)
        elif isinstance(v, list) and len(v) >= 1 and isinstance(v[0], dict):
            logger.info("{}{} : ".format(delimiter * " ",
                                        coloring(str(k), "HEADER")))
            for value in v:
                print_dict(value, logger, delimiter + 4)
        else:
            logger.info("{}{} : {}".format(delimiter * " ",
                                           coloring(k, "HEADER"),
                                           coloring(v, "OKGREEN")))
        if k.isupper():
            logger.info(placeholder)

def advertise(logger):
    """
    Show the advertising message like the following:
    ===========================================================
    ==        PaddleGAN is powered by PaddlePaddle !        ==
    ===========================================================
    ==                                                       ==
    ==   For more info please go to the following website.   ==
    ==                                                       ==
    ==       https://github.com/PaddlePaddle/PaddleGAN      ==
    ===========================================================
    """
    copyright = "PaddleGAN is powered by PaddlePaddle !"
    ad = "For more info please go to the following website."
    website = "https://github.com/PaddlePaddle/PaddleGAN"
    AD_LEN = 6 + len(max([copyright, ad, website], key=len))

    logger.info(
        coloring("\n{0}\n{1}\n{2}\n{3}\n{4}\n{5}\n{6}\n{7}\n".format(
            "=" * (AD_LEN + 4),
            "=={}==".format(copyright.center(AD_LEN)),
            "=" * (AD_LEN + 4),
            "=={}==".format(' ' * AD_LEN),
            "=={}==".format(ad.center(AD_LEN)),
            "=={}==".format(' ' * AD_LEN),
            "=={}==".format(website.center(AD_LEN)),
            "=" * (AD_LEN + 4), ), "RED"))

def print_config(config, logger):
    """
    visualize configs
    Arguments:
        config: configs
    """
    advertise(logger)
    print_dict(config, logger)

def override(dl, ks, v):
    """
    Recursively replace dict of list
    Args:
        dl(dict or list): dict or list to be replaced
        ks(list): list of keys
        v(str): value to be replaced
    """
    def str2num(v):
        try:
            return eval(v)
        except Exception:
            return v

    assert isinstance(dl, (list, dict)), ("{} should be a list or a dict")
    assert len(ks) > 0, ('lenght of keys should larger than 0')
    if isinstance(dl, list):
        k = str2num(ks[0])
        if len(ks) == 1:
            assert k < len(dl), ('index({}) out of range({})'.format(k, dl))
            dl[k] = str2num(v)
        else:
            override(dl[k], ks[1:], v)
    else:
        if len(ks) == 1:
            assert ks[0] in dl, ('{} is not exist in {}'.format(ks[0], dl))
            dl[ks[0]] = str2num(v)
        else:
            override(dl[ks[0]], ks[1:], v)


def override_config(config, options=None):
    """
    Recursively override the config
    Args:
        config(dict): dict to be replaced
        options(list): list of pairs(key0.key1.idx.key2=value)
            such as: [
                'topk=2',
                'VALID.transforms.1.ResizeImage.resize_short=300'
            ]
    Returns:
        config(dict): replaced config
    """
    if options is not None:
        for opt in options:
            assert isinstance(opt,
                              str), ("option({}) should be a str".format(opt))
            assert "=" in opt, (
                "option({}) should contain a ="
                "to distinguish between key and value".format(opt))
            pair = opt.split('=')
            assert len(pair) == 2, ("there can be only a = in the option")
            key, value = pair
            keys = key.split('.')
            override(config, keys, value)

    return config


def get_config(fname, overrides=None, show=False):
    """
    Read config from file
    """
    assert os.path.exists(fname), ('config file({}) is not exist'.format(fname))
    config = parse_config(fname)
    override_config(config, overrides)
    if show:
        logger = get_logger(name='ppgan.config')
        print_config(config, logger)

    return config


def cfg2dict(cfg):
    if isinstance(cfg, AttrDict):
        cfg = dict(cfg)
    for k in cfg.keys():
        if isinstance(cfg[k], AttrDict):
            cfg[k] = cfg2dict(cfg[k])
    return cfg
    