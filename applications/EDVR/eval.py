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
import paddle.fluid as fluid

from utils.config_utils import *
import models
from reader import get_reader
from metrics import get_metrics
from utils.utility import check_cuda
from utils.utility import check_version

logging.root.handlers = []
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
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
        '--batch_size',
        type=int,
        default=None,
        help='test batch size. None to use config file setting.')
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
        '--save_dir',
        type=str,
        default=os.path.join('data', 'evaluate_results'),
        help='output dir path, default to use ./data/evaluate_results')
    parser.add_argument(
        '--log_interval',
        type=int,
        default=1,
        help='mini-batch interval to log.')
    args = parser.parse_args()
    return args


def test(args):
    # parse config
    config = parse_config(args.config)
    test_config = merge_configs(config, 'test', vars(args))
    print_configs(test_config, "Test")

    # build model
    test_model = models.get_model(args.model_name, test_config, mode='test')
    test_model.build_input(use_dataloader=False)
    test_model.build_model()
    test_feeds = test_model.feeds()
    test_fetch_list = test_model.fetches()

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    exe.run(fluid.default_startup_program())

    if args.weights:
        assert os.path.exists(
            args.weights), "Given weight dir {} not exist.".format(args.weights)
    weights = args.weights or test_model.get_weights()

    logger.info('load test weights from {}'.format(weights))

    test_model.load_test_weights(exe, weights,
                                 fluid.default_main_program(), place)

    # get reader and metrics
    test_reader = get_reader(args.model_name.upper(), 'test', test_config)
    test_metrics = get_metrics(args.model_name.upper(), 'test', test_config)

    test_feeder = fluid.DataFeeder(place=place, feed_list=test_feeds)

    epoch_period = []
    for test_iter, data in enumerate(test_reader()):
        cur_time = time.time()
        if args.model_name == 'ETS':
            feat_data = [items[:3] for items in data]
            vinfo = [items[3:] for items in data]
            test_outs = exe.run(fetch_list=test_fetch_list,
                                feed=test_feeder.feed(feat_data),
                                return_numpy=False)
            test_outs += [vinfo]
        elif args.model_name == 'TALL':
            feat_data = [items[:2] for items in data]
            vinfo = [items[2:] for items in data]
            test_outs = exe.run(fetch_list=test_fetch_list,
                                feed=test_feeder.feed(feat_data),
                                return_numpy=True)
            test_outs += [vinfo]
        elif args.model_name == 'EDVR':
            #img_data = [item[0] for item in data]
            #gt_data = [item[1] for item in data]
            #gt_data = gt_data[0]
            #gt_data = np.transpose(gt_data, (1,2,0))
            #gt_data = gt_data[:, :, ::-1]
            #print('input', img_data)
            #print('gt', gt_data)
            feat_data = [items[:2] for items in data]
            print("feat_data[0] shape: ", feat_data[0][0].shape)
            exit()
            vinfo = [items[2:] for items in data]
            test_outs = exe.run(fetch_list=test_fetch_list,
                                feed=test_feeder.feed(feat_data),
                                return_numpy=True)
            #output = test_outs[1]
            #print('output', output)
            test_outs += [vinfo]
        else:
            test_outs = exe.run(fetch_list=test_fetch_list,
                                feed=test_feeder.feed(data))
        period = time.time() - cur_time
        epoch_period.append(period)
        test_metrics.accumulate(test_outs)

        # metric here
        if args.log_interval > 0 and test_iter % args.log_interval == 0:
            info_str = '[EVAL] Batch {}'.format(test_iter)
            test_metrics.calculate_and_log_out(test_outs, info_str)

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    test_metrics.finalize_and_log_out("[EVAL] eval finished. ", args.save_dir)


if __name__ == "__main__":
    args = parse_args()
    # check whether the installed paddle is compiled with GPU
    check_cuda(args.use_gpu)
    check_version()
    logger.info(args)

    test(args)
