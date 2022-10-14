#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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

import math
import sys
import paddle
from paddle.nn import functional as F
from paddle import autograd

from .base_model import BaseModel
from .builder import MODELS
from .generators.builder import build_generator
from .discriminators.builder import build_discriminator
from .criterions.builder import build_criterion
from ..modules.init import init_weights
from collections import OrderedDict
from ..solver import build_lr_scheduler, build_optimizer
from ppgan.utils.visual import *
from ppgan.models.generators.gfpganv1_arch import FacialComponentDiscriminator
from ppgan.utils.download import get_path_from_url


@MODELS.register()
class GFPGANModel(BaseModel):
    """ This class implements the gfpgan model.

    """
    def __init__(self, **opt):

        super(GFPGANModel, self).__init__()
        self.opt = opt
        train_opt = opt
        if 'image_visual' in self.opt['path']:
            self.image_paths = self.opt['path']['image_visual']
        self.current_iter = 0
        self.nets['net_g'] = build_generator(opt['network_g'])
        self.log_size = int(math.log(self.opt['network_g']['out_size'], 2))
        # define networks (both generator and discriminator)
        self.nets['net_g_ema'] = build_generator(self.opt['network_g'])
        self.nets['net_d'] = build_discriminator(self.opt['network_d'])
        self.nets['net_g_ema'].eval()
        pretrain_network_g = self.opt['path'].get('pretrain_network_g', None)
        if pretrain_network_g != None:
            t_weight = get_path_from_url(pretrain_network_g)
            t_weight = paddle.load(t_weight)
            if 'net_g' in t_weight:
                self.nets['net_g'].set_state_dict(t_weight['net_g'])
                self.nets['net_g_ema'].set_state_dict(t_weight['net_g_ema'])
            else:
                self.nets['net_g'].set_state_dict(t_weight)
                self.nets['net_g_ema'].set_state_dict(t_weight)

            del t_weight

        self.nets['net_d'].train()
        self.nets['net_g'].train()
        if ('network_d_left_eye' in self.opt
                and 'network_d_right_eye' in self.opt
                and 'network_d_mouth' in self.opt):
            self.use_facial_disc = True
        else:
            self.use_facial_disc = False

        if self.use_facial_disc:
            # left eye
            self.nets['net_d_left_eye'] = FacialComponentDiscriminator()
            self.nets['net_d_right_eye'] = FacialComponentDiscriminator()
            self.nets['net_d_mouth'] = FacialComponentDiscriminator()
            load_path = self.opt['path'].get('pretrain_network_d_left_eye')
            if load_path is not None:
                load_val = get_path_from_url(load_path)
                load_val = paddle.load(load_val)
                self.nets['net_d_left_eye'].set_state_dict(load_val)
                self.nets['net_d_right_eye'].set_state_dict(load_val)
                self.nets['net_d_mouth'].set_state_dict(load_val)
                del load_val
            self.nets['net_d_left_eye'].train()
            self.nets['net_d_right_eye'].train()
            self.nets['net_d_mouth'].train()
            self.cri_component = build_criterion(train_opt['gan_component_opt'])

        if train_opt.get('pixel_opt'):
            self.cri_pix = build_criterion(train_opt['pixel_opt'])
        else:
            self.cri_pix = None

        # perceptual loss
        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_criterion(train_opt['perceptual_opt'])
        else:
            self.cri_perceptual = None

        # L1 loss is used in pyramid loss, component style loss and identity loss
        self.cri_l1 = build_criterion(train_opt['L1_opt'])

        # gan loss (wgan)
        self.cri_gan = build_criterion(train_opt['gan_opt'])

        # ----------- define identity loss ----------- #
        if 'network_identity' in self.opt:
            self.use_identity = True
        else:
            self.use_identity = False

        if self.use_identity:
            # define identity network
            self.network_identity = build_discriminator(
                self.opt['network_identity'])
            load_path = self.opt['path'].get('pretrain_network_identity')
            if load_path is not None:
                load_val = get_path_from_url(load_path)
                load_val = paddle.load(load_val)
                self.network_identity.set_state_dict(load_val)
                del load_val
            self.network_identity.eval()
            for param in self.network_identity.parameters():
                param.stop_gradient = True

        # regularization weights
        self.r1_reg_weight = train_opt['r1_reg_weight']  # for discriminator
        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)
        self.net_d_reg_every = train_opt['net_d_reg_every']

    def setup_input(self, data):
        self.lq = data['lq']

        if 'gt' in data:
            self.gt = data['gt']

        if 'loc_left_eye' in data:
            # get facial component locations, shape (batch, 4)
            self.loc_left_eyes = data['loc_left_eye'].astype('float32')
            self.loc_right_eyes = data['loc_right_eye'].astype('float32')
            self.loc_mouths = data['loc_mouth'].astype('float32')

    def forward(self, test_mode=False, regularize=False):
        pass

    def train_iter(self, optimizers=None):
        # optimize nets['net_g']
        for p in self.nets['net_d'].parameters():
            p.stop_gradient = True
        self.optimizers['optim_g'].clear_grad(set_to_zero=False)

        # do not update facial component net_d
        if self.use_facial_disc:
            for p in self.nets['net_d_left_eye'].parameters():
                p.stop_gradient = True
            for p in self.nets['net_d_right_eye'].parameters():
                p.stop_gradient = True
            for p in self.nets['net_d_mouth'].parameters():
                p.stop_gradient = True

        # image pyramid loss weight
        pyramid_loss_weight = self.opt.get('pyramid_loss_weight', 0)
        if pyramid_loss_weight > 0 and self.current_iter > self.opt.get(
                'remove_pyramid_loss', float('inf')):
            pyramid_loss_weight = 1e-12  # very small weight to avoid unused param error
        if pyramid_loss_weight > 0:
            self.output, out_rgbs = self.nets['net_g'](self.lq, return_rgb=True)
            pyramid_gt = self.construct_img_pyramid()
        else:
            self.output, out_rgbs = self.nets['net_g'](self.lq,
                                                       return_rgb=False)

        # get roi-align regions
        if self.use_facial_disc:
            self.get_roi_regions(eye_out_size=80, mouth_out_size=120)
        l_g_total = 0
        if (self.current_iter % self.net_d_iters == 0
                and self.current_iter > self.net_d_init_iters):
            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, self.gt)
                l_g_total += l_g_pix
                self.losses['l_g_pix'] = l_g_pix

            # image pyramid loss
            if pyramid_loss_weight > 0:
                for i in range(0, self.log_size - 2):
                    l_pyramid = self.cri_l1(out_rgbs[i],
                                            pyramid_gt[i]) * pyramid_loss_weight
                    l_g_total += l_pyramid
                    self.losses[f'l_p_{2**(i+3)}'] = l_pyramid

            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(
                    self.output, self.gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    self.losses['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    self.losses['l_g_style'] = l_g_style

            # gan loss
            fake_g_pred = self.nets['net_d'](self.output)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            self.losses['l_g_gan'] = l_g_gan

            # facial component loss
            if self.use_facial_disc:
                # left eye
                fake_left_eye, fake_left_eye_feats = self.nets[
                    'net_d_left_eye'](self.left_eyes, return_feats=True)
                l_g_gan = self.cri_component(fake_left_eye, True, is_disc=False)
                l_g_total += l_g_gan
                self.losses['l_g_gan_left_eye'] = l_g_gan
                # right eye
                fake_right_eye, fake_right_eye_feats = self.nets[
                    'net_d_right_eye'](self.right_eyes, return_feats=True)
                l_g_gan = self.cri_component(fake_right_eye,
                                             True,
                                             is_disc=False)
                l_g_total += l_g_gan
                self.losses['l_g_gan_right_eye'] = l_g_gan
                # mouth
                fake_mouth, fake_mouth_feats = self.nets['net_d_mouth'](
                    self.mouths, return_feats=True)
                l_g_gan = self.cri_component(fake_mouth, True, is_disc=False)
                l_g_total += l_g_gan
                self.losses['l_g_gan_mouth'] = l_g_gan

                if self.opt.get('comp_style_weight', 0) > 0:
                    # get gt feat
                    _, real_left_eye_feats = self.nets['net_d_left_eye'](
                        self.left_eyes_gt, return_feats=True)
                    _, real_right_eye_feats = self.nets['net_d_right_eye'](
                        self.right_eyes_gt, return_feats=True)
                    _, real_mouth_feats = self.nets['net_d_mouth'](
                        self.mouths_gt, return_feats=True)

                    def _comp_style(feat, feat_gt, criterion):
                        return criterion(self._gram_mat(
                            feat[0]), self._gram_mat(
                                feat_gt[0].detach())) * 0.5 + criterion(
                                    self._gram_mat(feat[1]),
                                    self._gram_mat(feat_gt[1].detach()))

                    # facial component style loss
                    comp_style_loss = 0
                    comp_style_loss += _comp_style(fake_left_eye_feats,
                                                   real_left_eye_feats,
                                                   self.cri_l1)
                    comp_style_loss += _comp_style(fake_right_eye_feats,
                                                   real_right_eye_feats,
                                                   self.cri_l1)
                    comp_style_loss += _comp_style(fake_mouth_feats,
                                                   real_mouth_feats,
                                                   self.cri_l1)
                    comp_style_loss = comp_style_loss * self.opt[
                        'comp_style_weight']
                    l_g_total += comp_style_loss
                    self.losses['l_g_comp_style_loss'] = comp_style_loss

            # identity loss
            if self.use_identity:
                identity_weight = self.opt['identity_weight']
                # get gray images and resize
                out_gray = self.gray_resize_for_identity(self.output)
                gt_gray = self.gray_resize_for_identity(self.gt)

                identity_gt = self.network_identity(gt_gray).detach()
                identity_out = self.network_identity(out_gray)
                l_identity = self.cri_l1(identity_out,
                                         identity_gt) * identity_weight
                l_g_total += l_identity
                self.losses['l_identity'] = l_identity

            l_g_total.backward()
            self.optimizers['optim_g'].step()
        # EMA
        self.accumulate(self.nets['net_g_ema'],
                        self.nets['net_g'],
                        decay=0.5**(32 / (10 * 1000)))

        # ----------- optimize net_d ----------- #
        for p in self.nets['net_d'].parameters():
            p.stop_gradient = False
        self.optimizers['optim_d'].clear_grad(set_to_zero=False)
        if self.use_facial_disc:
            for p in self.nets['net_d_left_eye'].parameters():
                p.stop_gradient = False
            for p in self.nets['net_d_right_eye'].parameters():
                p.stop_gradient = False
            for p in self.nets['net_d_mouth'].parameters():
                p.stop_gradient = False
            self.optimizers['optim_net_d_left_eye'].clear_grad(
                set_to_zero=False)
            self.optimizers['optim_net_d_right_eye'].clear_grad(
                set_to_zero=False)
            self.optimizers['optim_net_d_mouth'].clear_grad(set_to_zero=False)
        fake_d_pred = self.nets['net_d'](self.output.detach())
        real_d_pred = self.nets['net_d'](self.gt)

        l_d = self.cri_gan(real_d_pred, True, is_disc=True) + self.cri_gan(
            fake_d_pred, False, is_disc=True)
        self.losses['l_d'] = l_d
        # In WGAN, real_score should be positive and fake_score should be negative
        self.losses['real_score'] = real_d_pred.detach().mean()
        self.losses['fake_score'] = fake_d_pred.detach().mean()
        l_d.backward()
        if self.current_iter % self.net_d_reg_every == 0:
            self.gt.stop_gradient = False
            real_pred = self.nets['net_d'](self.gt)
            l_d_r1 = r1_penalty(real_pred, self.gt)
            l_d_r1 = (self.r1_reg_weight / 2 * l_d_r1 * self.net_d_reg_every +
                      0 * real_pred[0])
            self.losses['l_d_r1'] = l_d_r1.detach().mean()
            l_d_r1.backward()

        self.optimizers['optim_d'].step()

        # optimize facial component discriminators
        if self.use_facial_disc:
            # left eye
            fake_d_pred, _ = self.nets['net_d_left_eye'](
                self.left_eyes.detach())
            real_d_pred, _ = self.nets['net_d_left_eye'](self.left_eyes_gt)
            l_d_left_eye = self.cri_component(
                real_d_pred, True, is_disc=True) + self.cri_gan(
                    fake_d_pred, False, is_disc=True)
            self.losses['l_d_left_eye'] = l_d_left_eye
            l_d_left_eye.backward()
            # right eye
            fake_d_pred, _ = self.nets['net_d_right_eye'](
                self.right_eyes.detach())
            real_d_pred, _ = self.nets['net_d_right_eye'](self.right_eyes_gt)
            l_d_right_eye = self.cri_component(
                real_d_pred, True, is_disc=True) + self.cri_gan(
                    fake_d_pred, False, is_disc=True)
            self.losses['l_d_right_eye'] = l_d_right_eye
            l_d_right_eye.backward()
            # mouth
            fake_d_pred, _ = self.nets['net_d_mouth'](self.mouths.detach())
            real_d_pred, _ = self.nets['net_d_mouth'](self.mouths_gt)
            l_d_mouth = self.cri_component(real_d_pred, True,
                                           is_disc=True) + self.cri_gan(
                                               fake_d_pred, False, is_disc=True)
            self.losses['l_d_mouth'] = l_d_mouth
            l_d_mouth.backward()

            self.optimizers['optim_net_d_left_eye'].step()
            self.optimizers['optim_net_d_right_eye'].step()
            self.optimizers['optim_net_d_mouth'].step()
        # if self.current_iter%1000==0:

    def test_iter(self, metrics=None):
        self.nets['net_g_ema'].eval()
        self.fake_img, _ = self.nets['net_g_ema'](self.lq)
        self.visual_items['cur_fake'] = self.fake_img[0]
        self.visual_items['cur_gt'] = self.gt[0]
        self.visual_items['cur_lq'] = self.lq[0]
        with paddle.no_grad():
            if metrics is not None:
                for metric in metrics.values():
                    metric.update(self.fake_img.detach().numpy(),
                                  self.gt.detach().numpy())

    def setup_lr_schedulers(self, cfg):
        self.lr_scheduler = OrderedDict()
        self.lr_scheduler['_g'] = build_lr_scheduler(cfg)
        self.lr_scheduler['_component'] = build_lr_scheduler(cfg)
        cfg_d = cfg.copy()
        net_d_reg_ratio = self.net_d_reg_every / (self.net_d_reg_every + 1)
        cfg_d['learning_rate'] *= net_d_reg_ratio
        self.lr_scheduler['_d'] = build_lr_scheduler(cfg_d)
        return self.lr_scheduler

    def setup_optimizers(self, lr, cfg):
        # ----------- optimizer g ----------- #
        net_g_reg_ratio = 1
        parameters = []
        parameters += self.nets['net_g'].parameters()
        cfg['optim_g']['beta1'] = 0**net_g_reg_ratio
        cfg['optim_g']['beta2'] = 0.99**net_g_reg_ratio

        self.optimizers['optim_g'] = build_optimizer(cfg['optim_g'],
                                                     self.lr_scheduler['_g'],
                                                     parameters)

        # ----------- optimizer d ----------- #
        net_d_reg_ratio = self.net_d_reg_every / (self.net_d_reg_every + 1)
        parameters = []
        parameters += self.nets['net_d'].parameters()
        cfg['optim_d']['beta1'] = 0**net_d_reg_ratio
        cfg['optim_d']['beta2'] = 0.99**net_d_reg_ratio

        self.optimizers['optim_d'] = build_optimizer(cfg['optim_d'],
                                                     self.lr_scheduler['_d'],
                                                     parameters)

        # ----------- optimizers for facial component networks ----------- #
        if self.use_facial_disc:
            parameters = []
            parameters += self.nets['net_d_left_eye'].parameters()

            self.optimizers['optim_net_d_left_eye'] = build_optimizer(
                cfg['optim_component'], self.lr_scheduler['_component'],
                parameters)

            parameters = []
            parameters += self.nets['net_d_right_eye'].parameters()

            self.optimizers['optim_net_d_right_eye'] = build_optimizer(
                cfg['optim_component'], self.lr_scheduler['_component'],
                parameters)

            parameters = []
            parameters += self.nets['net_d_mouth'].parameters()

            self.optimizers['optim_net_d_mouth'] = build_optimizer(
                cfg['optim_component'], self.lr_scheduler['_component'],
                parameters)

        return self.optimizers

    def construct_img_pyramid(self):
        """Construct image pyramid for intermediate restoration loss"""
        pyramid_gt = [self.gt]
        down_img = self.gt
        for _ in range(0, self.log_size - 3):
            down_img = F.interpolate(down_img,
                                     scale_factor=0.5,
                                     mode='bilinear',
                                     align_corners=False)
            pyramid_gt.insert(0, down_img)
        return pyramid_gt

    def get_roi_regions(self, eye_out_size=80, mouth_out_size=120):
        from paddle.vision.ops import roi_align
        face_ratio = int(self.opt['network_g']['out_size'] / 512)
        eye_out_size *= face_ratio
        mouth_out_size *= face_ratio

        rois_eyes = []
        rois_mouths = []
        num_eye = []
        num_mouth = []
        for b in range(self.loc_left_eyes.shape[0]):  # loop for batch size
            # left eye and right eye

            img_inds = paddle.ones([2, 1], dtype=self.loc_left_eyes.dtype) * b
            bbox = paddle.stack(
                [self.loc_left_eyes[b, :], self.loc_right_eyes[b, :]],
                axis=0)  # shape: (2, 4)
            # rois = paddle.concat([img_inds, bbox], axis=-1)  # shape: (2, 5)
            rois_eyes.append(bbox)
            # mouse
            img_inds = paddle.ones([1, 1], dtype=self.loc_left_eyes.dtype) * b
            num_eye.append(2)
            num_mouth.append(1)
            # rois = paddle.concat([img_inds, self.loc_mouths[b:b + 1, :]], axis=-1)  # shape: (1, 5)
            rois_mouths.append(self.loc_mouths[b:b + 1, :])
        rois_eyes = paddle.concat(rois_eyes, 0)
        rois_mouths = paddle.concat(rois_mouths, 0)
        # real images
        num_eye = paddle.to_tensor(num_eye, dtype='int32')
        num_mouth = paddle.to_tensor(num_mouth, dtype='int32')

        all_eyes = roi_align(self.gt,
                             boxes=rois_eyes,
                             boxes_num=num_eye,
                             output_size=eye_out_size,
                             aligned=False) * face_ratio
        self.left_eyes_gt = all_eyes[0::2, :, :, :]
        self.right_eyes_gt = all_eyes[1::2, :, :, :]
        self.mouths_gt = roi_align(self.gt,
                                   boxes=rois_mouths,
                                   boxes_num=num_mouth,
                                   output_size=mouth_out_size,
                                   aligned=False) * face_ratio
        # output
        all_eyes = roi_align(self.output,
                             boxes=rois_eyes,
                             boxes_num=num_eye,
                             output_size=eye_out_size,
                             aligned=False) * face_ratio
        self.left_eyes = all_eyes[0::2, :, :, :]
        self.right_eyes = all_eyes[1::2, :, :, :]
        self.mouths = roi_align(self.output,
                                boxes=rois_mouths,
                                boxes_num=num_mouth,
                                output_size=mouth_out_size,
                                aligned=False) * face_ratio

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (paddle.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            paddle.Tensor: Gram matrix.
        """
        n, c, h, w = x.shape
        features = x.reshape((n, c, w * h))
        features_t = features.transpose([0, 2, 1])
        gram = features.bmm(features_t) / (c * h * w)
        return gram

    def gray_resize_for_identity(self, out, size=128):
        out_gray = (0.2989 * out[:, 0, :, :] + 0.5870 * out[:, 1, :, :] +
                    0.1140 * out[:, 2, :, :])
        out_gray = out_gray.unsqueeze(1)
        out_gray = F.interpolate(out_gray, (size, size),
                                 mode='bilinear',
                                 align_corners=False)
        return out_gray

    def accumulate(self, model1, model2, decay=0.999):
        par1 = dict(model1.state_dict())
        par2 = dict(model2.state_dict())

        for k in par1.keys():
            par1[k] = par1[k] * decay + par2[k] * (1 - decay)

        model1.load_dict(par1)


def r1_penalty(real_pred, real_img):
    """R1 regularization for discriminator. The core idea is to
        penalize the gradient on real data alone: when the
        generator distribution produces the true data distribution
        and the discriminator is equal to 0 on the data manifold, the
        gradient penalty ensures that the discriminator cannot create
        a non-zero gradient orthogonal to the data manifold without
        suffering a loss in the GAN game.

        Ref:
        Eq. 9 in Which training methods for GANs do actually converge.
        """
    grad_real = paddle.grad(outputs=real_pred.sum(),
                            inputs=real_img,
                            create_graph=True)[0]
    grad_penalty = grad_real.pow(2).reshape(
        (grad_real.shape[0], -1)).sum(1).mean()
    return grad_penalty
