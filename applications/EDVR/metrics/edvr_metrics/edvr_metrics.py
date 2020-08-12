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

import numpy as np
import datetime
import logging
import json
import os
import cv2
import math

logger = logging.getLogger(__name__)

class MetricsCalculator():
    def __init__(
            self,
            name='EDVR',
            mode='train'):
        self.name = name
        self.mode = mode  # 'train', 'valid', 'test', 'infer'
        self.reset()
        self.total_frames = 9002 #100
        self.bolder_frames = 2

    def reset(self):
        logger.info('Resetting {} metrics...'.format(self.mode))
        if (self.mode == 'train') or (self.mode == 'valid'):
            self.aggr_loss = 0.0
        elif (self.mode == 'test') or (self.mode == 'infer'):
            self.result_dict = dict()

    def calculate_and_logout(self, fetch_list, info):
        pass

    def accumulate(self, fetch_list):
        loss = fetch_list[0]
        pred = fetch_list[1]
        gt = fetch_list[2]
        videoinfo = fetch_list[-1]
        print('videoinfo: ', videoinfo)
        videonames = [item[0] for item in videoinfo]
        framenames = [item[1] for item in videoinfo]
        for i in range(len(pred)):
            pred_i = pred[i]
            gt_i = gt[i]
            videoname_i = videonames[i]
            framename_i = framenames[i]
            if videoname_i not in self.result_dict.keys():
                self.result_dict[videoname_i] = {}
            if framename_i in self.result_dict[videoname_i].keys():
                logger.info("frame {} already processed in video {}, please check it".format(framename_i, videoname_i))
                raise
            is_bolder = (int(framename_i) > (self.total_frames - self.bolder_frames - 1)
                          or int(framename_i) < self.bolder_frames)
            psnr_i = get_psnr(pred_i, gt_i)
            img_i = get_img(pred_i)
            self.result_dict[videoname_i][framename_i] = [is_bolder, psnr_i]
            is_save = True
            if is_save and (i == len(pred) - 1):
                save_img(img_i,  framename_i)
            logger.info("video {}, frame {}, bolder {}, psnr = {}".format(videoname_i, framename_i, is_bolder, psnr_i))


    def finalize_metrics(self, savedir):
        avg_psnr = 0.
        avg_psnr_center = 0.
        avg_psnr_bolder = 0.
        center_num = 0.
        bolder_num = 0.
        for videoname in self.result_dict.keys():
            videoresult = self.result_dict[videoname]
            framelist = list(videoresult.keys())
            video_psnr_center = 0.
            video_psnr_bolder = 0.
            video_center_num = 0.
            video_bolder_num = 0.
            for frame in framelist:
                frameresult = videoresult[frame]
                is_bolder = frameresult[0]
                psnr = frameresult[1]
                if is_bolder:
                    video_bolder_num += 1
                    video_psnr_bolder += psnr
                else:
                    video_center_num += 1
                    video_psnr_center += psnr
            video_num = video_bolder_num + video_center_num
            video_psnr = video_psnr_center + video_psnr_bolder
            avg_psnr_bolder += video_psnr_bolder
            avg_psnr_center += video_psnr_center
            bolder_num += video_bolder_num
            center_num += video_center_num
            logger.info("video {}, total frame num/psnr {}/{}, center num/psnr {}/{}, bolder num/psnr {}/{}".format(
                               videoname, video_num, video_psnr/video_num,
                               video_center_num, video_psnr_center/video_center_num,
                               video_bolder_num, video_psnr_bolder/video_bolder_num))
        avg_psnr = avg_psnr_bolder + avg_psnr_center
        total_num = bolder_num + center_num
        avg_psnr = avg_psnr / total_num
        avg_psnr_center = avg_psnr_center / center_num
        avg_psnr_bolder = avg_psnr_bolder / bolder_num
        logger.info("Average psnr {}, center {}, bolder {}".format(avg_psnr, avg_psnr_center, avg_psnr_bolder))


def get_psnr(pred, gt):
    # pred and gt have range [0, 1]
    pred = pred.squeeze().astype(np.float64)
    pred = pred * 255.
    pred = pred.round()
    gt = gt.squeeze().astype(np.float64)
    gt = gt * 255.
    gt = gt.round()
    mse = np.mean((pred - gt)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


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

    
