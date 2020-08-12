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

import os
import sys
import time
import logging
import argparse
import ast
import numpy as np
try:
    import cPickle as pickle
except:
    import pickle
import paddle.fluid as fluid

from utils.config_utils import *
import models
from reader import get_reader
from metrics import get_metrics
from utils.utility import check_cuda

logging.root.handlers = []
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name',
        type=str,
        default='AttentionCluster',
        help='name of model to train.')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/attention_cluster.txt',
        help='path to config file of model')
    parser.add_argument(
        '--use_gpu',
        type=ast.literal_eval,
        default=True,
        help='default use gpu.')
    parser.add_argument(
        '--weights',
        type=str,
        default=None,
        help='weight path, None to automatically download weights provided by Paddle.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='sample number in a batch for inference.')
    parser.add_argument(
        '--save_dir',
        type=str,
        default='./',
        help='directory to store model and params file')
    args = parser.parse_args()
    return args


def save_inference_model(args):
    # parse config
    config = parse_config(args.config)
    infer_config = merge_configs(config, 'infer', vars(args))
    print_configs(infer_config, "Infer")
    infer_model = models.get_model(args.model_name, infer_config, mode='infer')
    infer_model.build_input(use_dataloader=False)
    infer_model.build_model()
    infer_feeds = infer_model.feeds()
    infer_outputs = infer_model.outputs()

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    if args.weights:
        assert os.path.exists(
            args.weights), "Given weight dir {} not exist.".format(args.weights)
    # if no weight files specified, download weights from paddle
    weights = args.weights or infer_model.get_weights()

    infer_model.load_test_weights(exe, weights,
                                  fluid.default_main_program(), place)

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    # saving inference model
    fluid.io.save_inference_model(
        args.save_dir,
        feeded_var_names=[item.name for item in infer_feeds],
        target_vars=infer_outputs,
        executor=exe,
        main_program=fluid.default_main_program(),
        model_filename=args.model_name + "_model.pdmodel",
        params_filename=args.model_name + "_params.pdparams")

    print("save inference model at %s" % (args.save_dir))


if __name__ == "__main__":
    args = parse_args()
    # check whether the installed paddle is compiled with GPU
    check_cuda(args.use_gpu)
    logger.info(args)

    save_inference_model(args)
