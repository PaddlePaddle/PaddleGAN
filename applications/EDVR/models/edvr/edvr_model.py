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
import time
import sys
import paddle.fluid as fluid
import math

def DCNPack(fea_and_offset, num_filters, kernel_size, stride=1, padding=0,
            dilation=1, deformable_groups=1, extra_offset_mask=True, name=None, local_lr=1.0):
    # create offset and mask similar to EDVR
    # To be added
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size, kernel_size]
    if name is None:
        conv_offset_mask_name = None
        dcn_name = None
    else:
        conv_offset_mask_name = name + '_conv_offset_mask'
        dcn_name = name + '_dcn'
    if extra_offset_mask:
        out = fluid.layers.conv2d(fea_and_offset[1], deformable_groups * 3 * kernel_size[0] * kernel_size[1],
                                  kernel_size, stride, padding, name = conv_offset_mask_name, 
                                  param_attr=fluid.ParamAttr(name=conv_offset_mask_name+'.w_0',
                                                             initializer=fluid.initializer.ConstantInitializer(value=0.0),
                                                             learning_rate=local_lr),
                                  bias_attr=fluid.ParamAttr(name=conv_offset_mask_name+'.b_0',
                                                            initializer=fluid.initializer.ConstantInitializer(value=0.0),
                                                            learning_rate=local_lr))
        x = fea_and_offset[0]
    else:
        x = fea_and_offset
        out = fluid.layers.conv2d(x, deformable_groups * 3 * kernel_size[0] * kernel_size[1],
                                  kernel_size, stride, padding, name = conv_offset_mask_name,
                                  param_attr=fluid.ParamAttr(name=conv_offset_mask_name+'.w_0',
                                                             initializer=fluid.initializer.Constant(value=0.0),
                                                             learning_rate=local_lr),
                                  bias_attr=fluid.ParamAttr(name=conv_offset_mask_name+'.b_0',
                                                            initializer=fluid.initializer.Constant(value=0.0),
                                                            learning_rate=local_lr))
    total_channels = deformable_groups * 3 * kernel_size[0] * kernel_size[1]
    split_channels = total_channels // 3
    o1 = out[:, 0:split_channels, :, :]
    o2 = out[:, split_channels:split_channels*2, :, :]
    mask = out[:, split_channels*2:, :, :]

    #o1 = out[:, 0::3, :, :]
    #o2 = out[:, 1::3, :, :]
    #mask = out[:, 2::3, :, :]

    offset = fluid.layers.concat([o1, o2], axis = 1)
    mask = fluid.layers.sigmoid(mask)

    #x = fluid.layers.Print(x, message='dcn_x')
    #offset = fluid.layers.Print(offset, message='dcn_offset')
    #mask = fluid.layers.Print(mask, message='dcn_mask')

    y = fluid.layers.deformable_conv(x, offset, mask, num_filters, kernel_size, stride=stride, padding=padding, 
                   deformable_groups=deformable_groups, modulated=True, name=dcn_name, im2col_step=1,
                   param_attr=fluid.ParamAttr(name=dcn_name+'.w_0', learning_rate=local_lr),
                   bias_attr=fluid.ParamAttr(name=dcn_name+'.b_0', learning_rate=local_lr))
    #y = fluid.layers.Print(y, message='dcn_y')
    return y


def PCD_Align(nbr_fea_l, ref_fea_l, nf=64, groups=8, local_lr=1.0):
    # L3
    L3_offset = fluid.layers.concat([nbr_fea_l[2], ref_fea_l[2]], axis = 1)
    #L3_offset = fluid.layers.Print(L3_offset, message='L3_offset1')
    L3_offset = fluid.layers.conv2d(L3_offset, nf, 3, stride=1, padding=1, name='PCD_Align_L3_offset_conv1', \
                  param_attr=fluid.ParamAttr(name='PCD_Align_L3_offset_conv1.w_0', learning_rate=local_lr),
                  bias_attr=fluid.ParamAttr(name='PCD_Align_L3_offset_conv1.b_0', learning_rate=local_lr))
    L3_offset = fluid.layers.leaky_relu(L3_offset, alpha=0.1)
    L3_offset = fluid.layers.conv2d(L3_offset, nf, 3, stride=1, padding=1, name='PCD_Align_L3_offset_conv2', \
                  param_attr=fluid.ParamAttr(name='PCD_Align_L3_offset_conv2.w_0', learning_rate=local_lr),
                  bias_attr=fluid.ParamAttr(name='PCD_Align_L3_offset_conv2.b_0', learning_rate=local_lr))
    L3_offset = fluid.layers.leaky_relu(L3_offset, alpha=0.1)

    #L3_offset = fluid.layers.Print(L3_offset, message="PCD_Align L3_offset", summarize=20)

    L3_fea = DCNPack([nbr_fea_l[2], L3_offset], nf, 3, stride=1, \
                     padding=1, deformable_groups = groups, name='PCD_Align_L3_dcn', local_lr=local_lr)

    #L3_offset = fluid.layers.Print(L3_offset, message='L3_offset2')
    L3_fea = fluid.layers.leaky_relu(L3_fea, alpha=0.1)
    #L3_fea = fluid.layers.Print(L3_fea, message="PCD_Align L3_fea", summarize=20)
    # L2
    L2_offset = fluid.layers.concat([nbr_fea_l[1], ref_fea_l[1]], axis = 1)
    L2_offset = fluid.layers.conv2d(L2_offset, nf, 3, stride=1, padding=1, name='PCD_Align_L2_offset_conv1',
                    param_attr=fluid.ParamAttr(name='PCD_Align_L2_offset_conv1.w_0', learning_rate=local_lr),
                    bias_attr=fluid.ParamAttr(name='PCD_Align_L2_offset_conv1.b_0', learning_rate=local_lr))
    L2_offset = fluid.layers.leaky_relu(L2_offset, alpha=0.1)
    L3_offset = fluid.layers.resize_bilinear(L3_offset, scale=2, align_corners=False, align_mode=0)
    #L3_offset = fluid.layers.Print(L3_offset, message='L3_offset3')
    L2_offset = fluid.layers.concat([L2_offset, L3_offset * 2], axis=1)
    L2_offset = fluid.layers.conv2d(L2_offset, nf, 3, stride=1, padding=1, name='PCD_Align_L2_offset_conv2',
                    param_attr=fluid.ParamAttr(name='PCD_Align_L2_offset_conv2.w_0', learning_rate=local_lr),
                    bias_attr=fluid.ParamAttr(name='PCD_Align_L2_offset_conv2.b_0', learning_rate=local_lr))
    L2_offset = fluid.layers.leaky_relu(L2_offset, alpha=0.1)
    L2_offset = fluid.layers.conv2d(L2_offset, nf, 3, stride=1, padding=1, name='PCD_Align_L2_offset_conv3',
                    param_attr=fluid.ParamAttr(name='PCD_Align_L2_offset_conv3.w_0', learning_rate=local_lr),
                    bias_attr=fluid.ParamAttr(name='PCD_Align_L2_offset_conv3.b_0', learning_rate=local_lr))
    L2_offset = fluid.layers.leaky_relu(L2_offset, alpha=0.1)
    L2_fea = DCNPack([nbr_fea_l[1], L2_offset], nf, 3, stride=1, \
                     padding=1, deformable_groups = groups, name='PCD_Align_L2_dcn', local_lr=local_lr)
    #L2_fea = fluid.layers.Print(L2_fea, message="L2_fea_after_dcn", summarize=20)
    L3_fea = fluid.layers.resize_bilinear(L3_fea, scale=2, align_corners=False, align_mode=0)
    #L3_fea = fluid.layers.Print(L3_fea, message="L3_fea_after_resize", summarize=20)
    L2_fea = fluid.layers.concat([L2_fea, L3_fea], axis=1)
    L2_fea = fluid.layers.conv2d(L2_fea, nf, 3, stride=1, padding=1, name='PCD_Align_L2_fea_conv',
                    param_attr=fluid.ParamAttr(name='PCD_Align_L2_fea_conv.w_0', learning_rate=local_lr),
                    bias_attr=fluid.ParamAttr(name='PCD_Align_L2_fea_conv.b_0', learning_rate=local_lr))
    L2_fea = fluid.layers.leaky_relu(L2_fea, alpha=0.1)
    # L1
    L1_offset = fluid.layers.concat([nbr_fea_l[0], ref_fea_l[0]], axis=1)
    L1_offset = fluid.layers.conv2d(L1_offset, nf, 3, stride=1, padding=1, name='PCD_Align_L1_offset_conv1',
                    param_attr=fluid.ParamAttr(name='PCD_Align_L1_offset_conv1.w_0', learning_rate=local_lr),
                    bias_attr=fluid.ParamAttr(name='PCD_Align_L1_offset_conv1.b_0', learning_rate=local_lr))
    L1_offset = fluid.layers.leaky_relu(L1_offset, alpha=0.1)
    L2_offset = fluid.layers.resize_bilinear(L2_offset, scale=2, align_corners=False, align_mode=0)
    L1_offset = fluid.layers.concat([L1_offset, L2_offset * 2], axis=1)
    L1_offset = fluid.layers.conv2d(L1_offset, nf, 3, stride=1, padding=1, name='PCD_Align_L1_offset_conv2',
                    param_attr=fluid.ParamAttr(name='PCD_Align_L1_offset_conv2.w_0', learning_rate=local_lr),
                    bias_attr=fluid.ParamAttr(name='PCD_Align_L1_offset_conv2.b_0', learning_rate=local_lr))
    L1_offset = fluid.layers.leaky_relu(L1_offset, alpha=0.1)
    L1_offset = fluid.layers.conv2d(L1_offset, nf, 3, stride=1, padding=1, name='PCD_Align_L1_offset_conv3',
                    param_attr=fluid.ParamAttr(name='PCD_Align_L1_offset_conv3.w_0', learning_rate=local_lr),
                    bias_attr=fluid.ParamAttr(name='PCD_Align_L1_offset_conv3.b_0', learning_rate=local_lr))
    L1_offset = fluid.layers.leaky_relu(L1_offset, alpha=0.1)
    L1_fea = DCNPack([nbr_fea_l[0], L1_offset], nf, 3, stride=1, padding=1, \
                     deformable_groups = groups, name='PCD_Align_L1_dcn', local_lr=local_lr) # this output is consistent
    #L1_fea = fluid.layers.Print(L1_fea, message="PCD_Align_L1_dcn", summarize=20)
    #L2_fea = fluid.layers.Print(L2_fea, message="L2_fea_before_resize", summarize=20)
    L2_fea = fluid.layers.resize_bilinear(L2_fea, scale=2, align_corners=False, align_mode=0)
    #L2_fea = fluid.layers.Print(L2_fea, message="L2_fea_after_resize", summarize=20)
    L1_fea = fluid.layers.concat([L1_fea, L2_fea], axis=1)
    L1_fea = fluid.layers.conv2d(L1_fea, nf, 3, stride=1, padding=1, name='PCD_Align_L1_fea_conv',
                    param_attr=fluid.ParamAttr(name='PCD_Align_L1_fea_conv.w_0', learning_rate=local_lr),
                    bias_attr=fluid.ParamAttr(name='PCD_Align_L1_fea_conv.b_0', learning_rate=local_lr))
    #L1_fea = fluid.layers.Print(L1_fea, message="PCD_Align_L1_fea_conv", summarize=20)
    # cascade
    offset = fluid.layers.concat([L1_fea, ref_fea_l[0]], axis=1)
    offset = fluid.layers.conv2d(offset, nf, 3, stride=1, padding=1, name='PCD_Align_cas_offset_conv1',
                    param_attr=fluid.ParamAttr(name='PCD_Align_cas_offset_conv1.w_0', learning_rate=local_lr),
                    bias_attr=fluid.ParamAttr(name='PCD_Align_cas_offset_conv1.b_0', learning_rate=local_lr))
    offset = fluid.layers.leaky_relu(offset, alpha=0.1)
    offset = fluid.layers.conv2d(offset, nf, 3, stride=1, padding=1, name='PCD_Align_cas_offset_conv2',
                    param_attr=fluid.ParamAttr(name='PCD_Align_cas_offset_conv2.w_0', learning_rate=local_lr),
                    bias_attr=fluid.ParamAttr(name='PCD_Align_cas_offset_conv2.b_0', learning_rate=local_lr))
    offset = fluid.layers.leaky_relu(offset, alpha=0.1)
    L1_fea = DCNPack([L1_fea, offset], nf, 3, stride=1, padding=1, \
                     deformable_groups = groups, name='PCD_Align_cascade_dcn', local_lr=local_lr) #this L1_fea is different
    L1_fea = fluid.layers.leaky_relu(L1_fea, alpha=0.1)
    #L1_fea = fluid.layers.Print(L1_fea, message="PCD_Align L1_fea output", summarize=20)

    return L1_fea


def TSA_Fusion(aligned_fea, nf=64, nframes=5, center=2):
    # In actual fact, nf == C should be required
    B, N, C, H, W = aligned_fea.shape

    # temporal fusion
    x_center = aligned_fea[:, center, :, :, :]
    emb_rf = fluid.layers.conv2d(x_center, nf, 3, stride=1, padding=1, name='tAtt_2')
    emb = fluid.layers.reshape(aligned_fea, [-1, C, H, W])
    emb = fluid.layers.conv2d(emb, nf, 3, stride=1, padding=1, name='tAtt_1')
    emb = fluid.layers.reshape(emb, [-1, N, nf, H, W])
    cor_l = []
    for i in range(N):
        emb_nbr = emb[:, i, :, :, :]
        cor_tmp = fluid.layers.reduce_sum(emb_nbr * emb_rf, dim=1, keep_dim = True)
        cor_l.append(cor_tmp)
    cor_prob = fluid.layers.concat(cor_l, axis=1)
    cor_prob = fluid.layers.sigmoid(cor_prob)
    cor_prob = fluid.layers.unsqueeze(cor_prob, axes=2)
    #cor_prob = fluid.layers.expand(cor_prob, [1, 1, C, 1, 1])
    cor_prob = fluid.layers.expand(cor_prob, [1, 1, nf, 1, 1])
    #cor_prob = fluid.layers.reshape(cor_prob, [-1, N*C, H, W])
    cor_prob = fluid.layers.reshape(cor_prob, [-1, N*nf, H, W])
    aligned_fea = fluid.layers.reshape(aligned_fea, [-1, N*C, H, W])
    aligned_fea = aligned_fea * cor_prob

    #aligned_fea = fluid.layers.Print(aligned_fea, message="aligned_fea temporal0", summarize=20) 


    fea = fluid.layers.conv2d(aligned_fea, nf, 1, stride=1, padding=0, name='fea_fusion')
    fea = fluid.layers.leaky_relu(fea, alpha=0.1)

    #fea = fluid.layers.Print(fea, message="aligned_fea temporal", summarize=20) 

    # spatial fusion
    att = fluid.layers.conv2d(aligned_fea, nf, 1, stride=1, padding=0, name='sAtt_1')
    att = fluid.layers.leaky_relu(att, alpha=0.1)
    #att = fluid.layers.Print(att, message="sAtt_1", summarize=20)
    att_max = fluid.layers.pool2d(att, pool_size=3, pool_stride=2, pool_padding=1, pool_type='max')
    att_avg = fluid.layers.pool2d(att, pool_size=3, pool_stride=2, pool_padding=1, pool_type='avg')
    att_pool = fluid.layers.concat([att_max, att_avg], axis=1)
    #att_pool = fluid.layers.Print(att_pool, message="att cat", summarize=20)
    att = fluid.layers.conv2d(att_pool, nf, 1, stride=1, padding=0, name='sAtt_2')
    #att = fluid.layers.Print(att, message="att sAtt2", summarize=20)
    att = fluid.layers.leaky_relu(att, alpha=0.1)

    #att = fluid.layers.Print(att, message="att spatial fusion", summarize=20)


    # pyramid
    att_L = fluid.layers.conv2d(att, nf, 1, stride=1, padding=0, name='sAtt_L1')
    att_L = fluid.layers.leaky_relu(att_L, alpha=0.1)

    #att_L = fluid.layers.Print(att_L, message="sAtt_L1", summarize=20)

    att_max = fluid.layers.pool2d(att_L, pool_size=3, pool_stride=2, pool_padding=1, pool_type='max')
    att_avg = fluid.layers.pool2d(att_L, pool_size=3, pool_stride=2, pool_padding=1, pool_type='avg')
    att_pool = fluid.layers.concat([att_max, att_avg], axis=1)
    att_L = fluid.layers.conv2d(att_pool, nf, 3, stride=1, padding=1, name='sAtt_L2')
    att_L = fluid.layers.leaky_relu(att_L, alpha=0.1)
    att_L = fluid.layers.conv2d(att_L, nf, 3, stride=1, padding=1, name='sAtt_L3')
    att_L = fluid.layers.leaky_relu(att_L, alpha=0.1)

    #att_L = fluid.layers.Print(att_L, message="att_L before resize", summarize=20)

    att_L = fluid.layers.resize_bilinear(att_L, scale=2, align_corners=False, align_mode=0)

    #att_L = fluid.layers.Print(att_L, message="att_L", summarize=20)

    att = fluid.layers.conv2d(att, nf, 3, stride=1, padding=1, name='sAtt_3')
    att = fluid.layers.leaky_relu(att, alpha=0.1)
    att = att + att_L
    att = fluid.layers.conv2d(att, nf, 1, stride=1, padding=0, name='sAtt_4')
    att = fluid.layers.leaky_relu(att, alpha=0.1)
    att = fluid.layers.resize_bilinear(att, scale=2, align_corners=False, align_mode=0)
    att = fluid.layers.conv2d(att, nf, 3, stride=1, padding=1, name='sAtt_5')
    att_add = fluid.layers.conv2d(att, nf, 1, stride=1, padding=0, name='sAtt_add_1')
    att_add = fluid.layers.leaky_relu(att_add, alpha=0.1)
    att_add = fluid.layers.conv2d(att_add, nf, 1, stride=1, padding=0, name='sAtt_add_2')
    att = fluid.layers.sigmoid(att)

    #att = fluid.layers.Print(att, message="att", summarize=20)

    fea = fea * att * 2 + att_add
    return fea

def get_initializer(fan_in, scale=0.1):
    std = math.sqrt(2.0/fan_in) * scale
    return fluid.initializer.NormalInitializer(loc=0.0, scale=std)

def ResidualBlock_noBN(x, nf=64, name='', local_lr=1.0):
    # in the pytorch code, conv1 and conv2 are initialized with scale factor 0.1 than MSRA,
    # thi will be added later.
    fan_in = x.shape[1] * 3 * 3
    out = fluid.layers.conv2d(x, nf, 3, stride=1, padding=1, name=name+'_conv1', act='relu',
                             param_attr = fluid.ParamAttr(initializer=get_initializer(fan_in, scale=0.1),
                                                          learning_rate=local_lr),
                             bias_attr=fluid.ParamAttr(learning_rate=local_lr))
    fan_in = out.shape[1] * 3 * 3
    out = fluid.layers.conv2d(out, nf, 3, stride=1, padding=1, name=name+'_conv2',
                             param_attr = fluid.ParamAttr(initializer=get_initializer(fan_in, scale=0.1),
                                                          learning_rate=local_lr),
                             bias_attr=fluid.ParamAttr(learning_rate=local_lr))
    return out + x


def MakeMultiBlocks(x, func, num_layers, nf=64, name='', local_lr=1.0):
    for i in range(num_layers):
        x = func(x, nf=nf, name=name+"_block%d"%i, local_lr=local_lr)
        #x = fluid.layers.Print(x, message=name+"_block%d"%i, summarize=20)
    return x


def EDVRArch(x, nf=64, nframes=5, groups=8, front_RBs=5, back_RBs=10, center=None,
             predeblur=False, HR_in=False, w_TSA=True, TSA_only=False):
    B, N, C, H, W = x.shape
    center = nframes//2 if center is None else center
    x_center = x[:, center, :, :, :]
    local_lr = 1.0 - float(TSA_only)
    if predeblur:
        # not implemented yet
        pass
    else:
        if HR_in:
            L1_fea = fluid.layers.reshape(x, [-1, C, H, W])
            L1_fea = fluid.layers.conv2d(L1_fea, nf, 3, stride=1, padding=1, name='conv_first_1',
                                         param_attr=fluid.ParamAttr(learning_rate=local_lr),
                                         bias_attr=fluid.ParamAttr(learning_rate=local_lr))
            L1_fea = fluid.layers.leaky_relu(L1_fea, alpha=0.1)
            L1_fea = fluid.layers.conv2d(L1_fea, nf, 3, stride=2, padding=1, name='conv_first_2',
                                         param_attr=fluid.ParamAttr(learning_rate=local_lr),
                                         bias_attr=fluid.ParamAttr(learning_rate=local_lr))
            L1_fea = fluid.layers.leaky_relu(L1_fea, alpha=0.1)
            L1_fea = fluid.layers.conv2d(L1_fea, nf, 3, stride=2, padding=1, name='conv_first_3',
                                         param_attr=fluid.ParamAttr(learning_rate=local_lr),
                                         bias_attr=fluid.ParamAttr(learning_rate=local_lr))
            L1_fea = fluid.layers.leaky_relu(L1_fea, alpha=0.1)
            H = H // 4
            W = W // 4
        else:
            L1_fea = fluid.layers.reshape(x, [-1, C, H, W])
            L1_fea = fluid.layers.conv2d(L1_fea, nf, 3, stride=1, padding=1, name='conv_first',
                                         param_attr=fluid.ParamAttr(learning_rate=local_lr),
                                         bias_attr=fluid.ParamAttr(learning_rate=local_lr))
            L1_fea = fluid.layers.leaky_relu(L1_fea, alpha=0.1)

    #L1_fea = fluid.layers.Print(L1_fea, message="L1_fea", summarize=20)

    # feature extraction
    L1_fea = MakeMultiBlocks(L1_fea, ResidualBlock_noBN, front_RBs, nf=nf, name='feature_extractor', local_lr = local_lr)
    # L2
    L2_fea = fluid.layers.conv2d(L1_fea, nf, 3, stride=2, padding=1, name='fea_L2_conv1',
                                 param_attr=fluid.ParamAttr(learning_rate=local_lr),
                                 bias_attr=fluid.ParamAttr(learning_rate=local_lr))
    L2_fea = fluid.layers.leaky_relu(L2_fea, alpha=0.1)
    L2_fea = fluid.layers.conv2d(L2_fea, nf, 3, stride=1, padding=1, name='fea_L2_conv2',
                                 param_attr=fluid.ParamAttr(learning_rate=local_lr),
                                 bias_attr=fluid.ParamAttr(learning_rate=local_lr))
    L2_fea = fluid.layers.leaky_relu(L2_fea, alpha=0.1)
    # L3
    L3_fea = fluid.layers.conv2d(L2_fea, nf, 3, stride=2, padding=1, name='fea_L3_conv1',
                                 param_attr=fluid.ParamAttr(learning_rate=local_lr),
                                 bias_attr=fluid.ParamAttr(learning_rate=local_lr))
    L3_fea = fluid.layers.leaky_relu(L3_fea, alpha=0.1)
    L3_fea = fluid.layers.conv2d(L3_fea, nf, 3, stride=1, padding=1, name='fea_L3_conv2',
                                 param_attr=fluid.ParamAttr(learning_rate=local_lr),
                                 bias_attr=fluid.ParamAttr(learning_rate=local_lr))
    L3_fea = fluid.layers.leaky_relu(L3_fea, alpha=0.1)

    L1_fea = fluid.layers.reshape(L1_fea, [-1, N, nf, H, W])
    L2_fea = fluid.layers.reshape(L2_fea, [-1, N, nf, H//2, W//2])
    L3_fea = fluid.layers.reshape(L3_fea, [-1, N, nf, H//4, W//4])

    #L3_fea = fluid.layers.Print(L3_fea, message="L3_fea", summarize=20)

    # pcd align
    # ref_feature_list
    ref_fea_l = [L1_fea[:, center, :, :, :], L2_fea[:, center, :, :, :], L3_fea[:, center, :, :, :]]
    aligned_fea = []
    for i in range(N):
        nbr_fea_l = [L1_fea[:, i, :, :, :], L2_fea[:, i, :, :, :], L3_fea[:, i, :, :, :]]
        aligned_fea.append(PCD_Align(nbr_fea_l, ref_fea_l, nf=nf, groups=groups, local_lr=local_lr))
    if w_TSA:
        aligned_fea = fluid.layers.stack(aligned_fea, axis=1) # [B, N, C, H, W]
        #aligned_fea = fluid.layers.Print(aligned_fea, message="aligned_fea", summarize=20)
        fea = TSA_Fusion(aligned_fea, nf=nf, nframes=nframes, center=center)
    else:
        aligned_fea = fluid.layers.concat(aligned_fea, axis=1) # [B, NC, H, W]
        #aligned_fea = fluid.layers.Print(aligned_fea, message="aligned_fea", summarize=20)
        fea = fluid.layers.conv2d(aligned_fea, nf, 1, stride=1, padding=0, name='tsa_fusion',
                                  param_attr=fluid.ParamAttr(learning_rate=local_lr),
                                  bias_attr=fluid.ParamAttr(learning_rate=local_lr))


    # reconstructor
    out = MakeMultiBlocks(fea, ResidualBlock_noBN, back_RBs, nf=nf, name='reconstructor', local_lr=local_lr)
    #out = fluid.layers.Print(out, message="multiblocks_reconstructor", summarize=20)
    out = fluid.layers.conv2d(out, nf*4, 3, stride=1, padding=1, name='upconv1',
                              param_attr=fluid.ParamAttr(learning_rate=local_lr),
                              bias_attr=fluid.ParamAttr(learning_rate=local_lr))
    out = fluid.layers.pixel_shuffle(out, 2)
    out = fluid.layers.leaky_relu(out, alpha=0.1)
    out = fluid.layers.conv2d(out, 64*4, 3, stride=1, padding=1, name='upconv2',
                              param_attr=fluid.ParamAttr(learning_rate=local_lr),
                              bias_attr=fluid.ParamAttr(learning_rate=local_lr))
    out = fluid.layers.pixel_shuffle(out, 2)
    out = fluid.layers.leaky_relu(out, alpha=0.1)
    out = fluid.layers.conv2d(out, 64, 3, stride=1, padding=1, name='HRconv',
                              param_attr=fluid.ParamAttr(learning_rate=local_lr),
                              bias_attr=fluid.ParamAttr(learning_rate=local_lr))
    out = fluid.layers.leaky_relu(out, alpha=0.1)
    out = fluid.layers.conv2d(out, 3, 3, stride=1, padding=1, name='conv_last',
                              param_attr=fluid.ParamAttr(learning_rate=local_lr),
                              bias_attr=fluid.ParamAttr(learning_rate=local_lr))
    if HR_in:
        base = x_center
    else:
        base = fluid.layers.resize_bilinear(x_center, scale=4, align_corners=False, align_mode=0)
    out += base
    #out = fluid.layers.Print(out, message="network output", summarize=20)
    return out

class EDVRModel(object):
    def __init__(self, nf=64, nframes=5, groups=8, front_RBs=5, back_RBs=10, center=None,
             predeblur=False, HR_in=False, w_TSA=True, TSA_only=False, mode='train'):
        self.nf = nf
        self.nframes = nframes
        self.groups = groups
        self.front_RBs = front_RBs
        self.back_RBs = back_RBs
        self.center = center
        self.predeblur = predeblur
        self.HR_in = HR_in
        self.w_TSA = w_TSA
        self.mode = mode
        self.TSA_only = TSA_only

    def net(self, x):
        return EDVRArch(x, nf = self.nf, nframes = self.nframes, 
                  groups = self.groups, 
                  front_RBs = self.front_RBs, 
                  back_RBs = self.back_RBs, 
                  center = self.center, 
                  predeblur = self.predeblur, 
                  HR_in = self.HR_in, 
                  w_TSA = self.w_TSA,
                  TSA_only = self.TSA_only)

