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
import cv2

from utils.config_utils import *
import models
from reader import get_reader
from metrics import get_metrics
from utils.utility import check_cuda
from utils.utility import check_version

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
   # parser.add_argument(
   #     '--weights',
   #     type=str,
   #     default=None,
   #     help='weight path, None to automatically download weights provided by Paddle.'
   # )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='sample number in a batch for inference.')
    parser.add_argument(
        '--filelist',
        type=str,
        default=None,
        help='path to inferenece data file lists file.')
    parser.add_argument(
        '--log_interval',
        type=int,
        default=1,
        help='mini-batch interval to log.')
    parser.add_argument(
        '--infer_topk',
        type=int,
        default=20,
        help='topk predictions to restore.')
    parser.add_argument(
        '--save_dir',
        type=str,
        default=os.path.join('data', 'predict_results'),
        help='directory to store results')
    parser.add_argument(
        '--video_path',
        type=str,
        default=None,
        help='directory to store results')
    args = parser.parse_args()
    return args

def get_img(pred):
    print('pred shape', pred.shape)
    pred = pred.squeeze()
    pred = np.clip(pred, a_min=0., a_max=1.0)
    pred = pred * 255
    pred = pred.round()
    pred = pred.astype('uint8')
    pred = np.transpose(pred, (1, 2, 0)) # chw -> hwc
    pred = pred[:, :, ::-1] # rgb -> bgr
    return pred

def save_img(img, framename):
    dirname = './demo/resultpng'
    filename = os.path.join(dirname, framename+'.png')
    cv2.imwrite(filename, img)


def infer(args):
    # parse config
    config = parse_config(args.config)
    infer_config = merge_configs(config, 'infer', vars(args))
    print_configs(infer_config, "Infer")
    
    model_path = '/workspace/video_test/video/for_eval/data/inference_model'
    model_filename = 'EDVR_model.pdmodel'
    params_filename = 'EDVR_params.pdparams'
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    [inference_program, feed_list, fetch_list] = fluid.io.load_inference_model(dirname=model_path, model_filename=model_filename, params_filename=params_filename, executor=exe)

    infer_reader = get_reader(args.model_name.upper(), 'infer', infer_config)
    #infer_metrics = get_metrics(args.model_name.upper(), 'infer', infer_config)
    #infer_metrics.reset()

    periods = []
    cur_time = time.time()
    for infer_iter, data in enumerate(infer_reader()):
        if args.model_name == 'EDVR':
            data_feed_in = [items[0] for items in data]
            video_info = [items[1:] for items in data]
            infer_outs = exe.run(inference_program,
                                 fetch_list=fetch_list,
                                 feed={feed_list[0]:np.array(data_feed_in)})
            infer_result_list = [item for item in infer_outs]
            videonames = [item[0] for item in video_info]
            framenames = [item[1] for item in video_info]
            for i in range(len(infer_result_list)):
                img_i = get_img(infer_result_list[i])
                save_img(img_i, 'img' + videonames[i] + framenames[i])
                
                

        prev_time = cur_time
        cur_time = time.time()
        period = cur_time - prev_time
        periods.append(period)

        #infer_metrics.accumulate(infer_result_list)

        if args.log_interval > 0 and infer_iter % args.log_interval == 0:
            logger.info('Processed {} samples'.format(infer_iter + 1))

    logger.info('[INFER] infer finished. average time: {}'.format(np.mean(periods)))

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    #infer_metrics.finalize_and_log_out(savedir=args.save_dir)


if __name__ == "__main__":
    args = parse_args()
    # check whether the installed paddle is compiled with GPU
    check_cuda(args.use_gpu)
    check_version()
    logger.info(args)

    infer(args)
