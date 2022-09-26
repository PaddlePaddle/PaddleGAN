#Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

from collections import namedtuple
import numpy as np

import paddle
import paddle.nn as nn
from paddle.utils.download import get_weights_path_from_url
from ..modules import init
from ..models.criterions.perceptual_loss import PerceptualVGG
from .builder import METRICS

lpips = True

VGG16_TORCHVISION_URL = 'https://paddlegan.bj.bcebos.com/models/vgg16_official.pdparams'
LINS_01_VGG_URL = 'https://paddlegan.bj.bcebos.com/models/lins_0.1_vgg.pdparams'


@METRICS.register()
class LPIPSMetric(paddle.metric.Metric):
    """Calculate LPIPS (Learned Perceptual Image Patch Similarity).

    Ref: https://arxiv.org/abs/1801.03924

    Args:
        net (str): Type of backbone net. Default: 'vgg'.
        version (str): Version of lpips method. Defalut: '0.1'.
        mean (list): Sequence of means for each channel of input image. Default: None.
        std (list): Sequence of standard deviations for each channel of input image. Default: None.

    Returns:
        float: lpips result.
    """
    def __init__(self, net='vgg', version='0.1', mean=None, std=None):
        self.net = net
        self.version = version

        self.loss_fn = LPIPS(net=net, version=version)

        if mean is None:
            self.mean = [0.5, 0.5, 0.5]
        else:
            self.mean = mean

        if std is None:
            self.std = [0.5, 0.5, 0.5]
        else:
            self.std = std

        self.reset()

    def reset(self):
        self.results = []

    def update(self, preds, gts):
        if not isinstance(preds, (list, tuple)):
            preds = [preds]

        if not isinstance(gts, (list, tuple)):
            gts = [gts]

        for pred, gt in zip(preds, gts):
            pred, gt = pred.astype(np.float32) / 255., gt.astype(
                np.float32) / 255.
            pred = paddle.vision.transforms.normalize(pred.transpose([2, 0, 1]),
                                                      self.mean, self.std)
            gt = paddle.vision.transforms.normalize(gt.transpose([2, 0, 1]),
                                                    self.mean, self.std)

            with paddle.no_grad():
                value = self.loss_fn(
                    paddle.to_tensor(pred).unsqueeze(0),
                    paddle.to_tensor(gt).unsqueeze(0))

                self.results.append(value.item())

    def accumulate(self):
        if paddle.distributed.get_world_size() > 1:
            results = paddle.to_tensor(self.results)
            results_list = []
            paddle.distributed.all_gather(results_list, results)
            self.results = paddle.concat(results_list).numpy()

        if len(self.results) <= 0:
            return 0.
        return np.mean(self.results)

    def name(self):
        return 'LPIPS'


def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2, 3], keepdim=keepdim)


# assumes scale factor is same for H and W
def upsample(in_tens, out_HW=(64, 64)):
    in_H, in_W = in_tens.shape[2], in_tens.shape[3]
    scale_factor_H, scale_factor_W = 1. * out_HW[0] / in_H, 1. * out_HW[1] / in_W

    return nn.Upsample(scale_factor=(scale_factor_H, scale_factor_W),
                       mode='bilinear',
                       align_corners=False)(in_tens)


def normalize_tensor(in_feat, eps=1e-10):
    norm_factor = paddle.sqrt(paddle.sum(in_feat**2, 1, keepdim=True))
    return in_feat / (norm_factor + eps)


# Learned perceptual metric
class LPIPS(nn.Layer):
    def __init__(self,
                 pretrained=True,
                 net='vgg',
                 version='0.1',
                 lpips=True,
                 spatial=False,
                 pnet_rand=False,
                 pnet_tune=False,
                 use_dropout=True,
                 model_path=None,
                 eval_mode=True,
                 verbose=True):
        # lpips - [True] means with linear calibration on top of base network
        # pretrained - [True] means load linear weights

        super(LPIPS, self).__init__()
        if (verbose):
            print(
                'Setting up [%s] perceptual loss: trunk [%s], v[%s], spatial [%s]'
                % ('LPIPS' if lpips else 'baseline', net, version,
                   'on' if spatial else 'off'))

        self.pnet_type = net
        self.pnet_tune = pnet_tune
        self.pnet_rand = pnet_rand
        self.spatial = spatial

        # false means baseline of just averaging all layers
        self.lpips = lpips

        self.version = version
        self.scaling_layer = ScalingLayer()

        if (self.pnet_type in ['vgg', 'vgg16']):
            net_type = vgg16
            self.chns = [64, 128, 256, 512, 512]
        elif (self.pnet_type == 'alex'):
            raise TypeError('alex not support now!')

        elif (self.pnet_type == 'squeeze'):
            raise TypeError('squeeze not support now!')

        self.L = len(self.chns)

        self.net = net_type(pretrained=True, requires_grad=False)

        if (lpips):
            self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
            self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
            self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
            self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
            self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
            self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]

            # TODO: add alex and squeezenet
            # 7 layers for squeezenet
            self.lins = nn.LayerList(self.lins)
            if (pretrained):
                if (model_path is None):
                    model_path = get_weights_path_from_url(LINS_01_VGG_URL)

                if (verbose):
                    print('Loading model from: %s' % model_path)

                self.lins.set_state_dict(paddle.load(model_path))

        if (eval_mode):
            self.eval()

    def forward(self, in0, in1, retPerLayer=False, normalize=False):
        # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
        if normalize:
            in0 = 2 * in0 - 1
            in1 = 2 * in1 - 1

        # v0.0 - original release had a bug, where input was not scaled
        in0_input, in1_input = (
            self.scaling_layer(in0),
            self.scaling_layer(in1)) if self.version == '0.1' else (in0, in1)
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}

        for kk in range(self.L):
            feats0[kk], feats1[kk] = normalize_tensor(
                outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk])**2

        if (self.lpips):
            if (self.spatial):
                res = [
                    upsample(self.lins[kk].model(diffs[kk]),
                             out_HW=in0.shape[2:]) for kk in range(self.L)
                ]
            else:
                res = [
                    spatial_average(self.lins[kk].model(diffs[kk]),
                                    keepdim=True) for kk in range(self.L)
                ]
        else:
            if (self.spatial):
                res = [
                    upsample(diffs[kk].sum(dim=1, keepdim=True),
                             out_HW=in0.shape[2:]) for kk in range(self.L)
                ]
            else:
                res = [
                    spatial_average(diffs[kk].sum(dim=1, keepdim=True),
                                    keepdim=True) for kk in range(self.L)
                ]

        val = res[0]
        for l in range(1, self.L):
            val += res[l]

        if (retPerLayer):
            return (val, res)
        else:
            return val


class ScalingLayer(nn.Layer):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer(
            'shift',
            paddle.to_tensor([-.030, -.088, -.188]).reshape([1, 3, 1, 1]))
        self.register_buffer(
            'scale',
            paddle.to_tensor([.458, .448, .450]).reshape([1, 3, 1, 1]))

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Layer):
    ''' A single linear layer which does a 1x1 conv '''
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()

        layers = [
            nn.Dropout(),
        ] if (use_dropout) else []
        layers += [
            nn.Conv2D(chn_in, chn_out, 1, stride=1, padding=0, bias_attr=False),
        ]
        self.model = nn.Sequential(*layers)


class vgg16(nn.Layer):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        self.vgg16 = PerceptualVGG(['3', '8', '15', '22', '29'], 'vgg16', False,
                                   VGG16_TORCHVISION_URL)

        if not requires_grad:
            for param in self.parameters():
                param.trainable = False

    def forward(self, x):
        out = self.vgg16(x)
        vgg_outputs = namedtuple(
            "VggOutputs",
            ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(out['3'], out['8'], out['15'], out['22'], out['29'])

        return out
