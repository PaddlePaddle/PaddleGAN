#!/usr/bin/python
# -*- encoding: utf-8 -*-
import paddle
from paddle import nn
import paddle.nn.functional as F

from paddle.utils.download import get_weights_path_from_url
import numpy as np

from .resnet import resnet18


class ConvBNReLU(paddle.nn.Layer):
    def __init__(self,
                 in_chan,
                 out_chan,
                 ks=3,
                 stride=1,
                 padding=1,
                 *args,
                 **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                              out_chan,
                              kernel_size=ks,
                              stride=stride,
                              padding=padding,
                              bias_attr=False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU()
        #self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class BiSeNetOutput(paddle.nn.Layer):
    def __init__(self, in_chan, mid_chan, n_classes, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan,
                                  n_classes,
                                  kernel_size=1,
                                  bias_attr=False)
        #self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x


class AttentionRefinementModule(paddle.nn.Layer):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan,
                                    out_chan,
                                    kernel_size=1,
                                    bias_attr=False)
        self.bn_atten = nn.BatchNorm(out_chan)
        self.sigmoid_atten = nn.Sigmoid()
        #self.init_weight()

    def forward(self, x):
        feat = self.conv(x)
        #atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = F.avg_pool2d(feat, feat.shape[2:])
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = feat * atten
        return out

    #def init_weight(self):
    #    for ly in self.children():
    #        if isinstance(ly, nn.Conv2d):
    #            nn.init.kaiming_normal_(ly.weight, a=1)
    #            if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class ContextPath(paddle.nn.Layer):
    def __init__(self, *args, **kwargs):
        super(ContextPath, self).__init__()
        self.resnet = resnet18()
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)

        #self.init_weight()

    def forward(self, x):
        H0, W0 = x.shape[2:]
        feat8, feat16, feat32 = self.resnet(x)
        H8, W8 = feat8.shape[2:]
        H16, W16 = feat16.shape[2:]
        H32, W32 = feat32.shape[2:]
        print('feat32.shape: ', feat32.shape[2:])

        avg = F.avg_pool2d(feat32, feat32.shape[2:])
        avg = self.conv_avg(avg)
        #avg_up = F.interpolate(avg, (H32, W32), mode='nearest')
        avg_up = F.resize_nearest(avg, out_shape=(H32, W32))

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg_up
        #feat32_up = F.interpolate(feat32_sum, (H16, W16), mode='nearest')
        feat32_up = F.resize_nearest(feat32_sum, out_shape=(H16, W16))
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        #feat16_up = F.interpolate(feat16_sum, (H8, W8), mode='nearest')
        feat16_up = F.resize_nearest(feat16_sum, out_shape=(H8, W8))
        feat16_up = self.conv_head16(feat16_up)

        return feat8, feat16_up, feat32_up  # x8, x8, x16


### This is not used, since I replace this with the resnet feature with the same size
class SpatialPath(paddle.nn.Layer):
    def __init__(self, *args, **kwargs):
        super(SpatialPath, self).__init__()
        self.conv1 = ConvBNReLU(3, 64, ks=7, stride=2, padding=3)
        self.conv2 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv3 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv_out = ConvBNReLU(64, 128, ks=1, stride=1, padding=0)
        #self.init_weight()

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        feat = self.conv_out(feat)
        return feat


class FeatureFusionModule(paddle.nn.Layer):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan,
                               out_chan // 4,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias_attr=False)
        self.conv2 = nn.Conv2d(out_chan // 4,
                               out_chan,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias_attr=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, fsp, fcp):
        fcat = paddle.concat([fsp, fcp], axis=1)
        feat = self.convblk(fcat)
        #atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = F.avg_pool2d(feat, feat.shape[2:])
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = feat * atten
        feat_out = feat_atten + feat
        return feat_out


class BiSeNet(paddle.nn.Layer):
    def __init__(self, n_classes, *args, **kwargs):
        super(BiSeNet, self).__init__()
        self.cp = ContextPath()
        ## here self.sp is deleted
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, n_classes)
        self.conv_out16 = BiSeNetOutput(128, 64, n_classes)
        self.conv_out32 = BiSeNetOutput(128, 64, n_classes)

    def forward(self, x):
        H, W = x.shape[2:]
        feat_res8, feat_cp8, feat_cp16 = self.cp(
            x)  # here return res3b1 feature
        feat_sp = feat_res8  # use res3b1 feature to replace spatial path feature
        feat_fuse = self.ffm(feat_sp, feat_cp8)

        feat_out = self.conv_out(feat_fuse)
        feat_out16 = self.conv_out16(feat_cp8)
        feat_out32 = self.conv_out32(feat_cp16)

        feat_out = F.resize_bilinear(feat_out, out_shape=(H, W))
        feat_out16 = F.resize_bilinear(feat_out16, out_shape=(H, W))
        feat_out32 = F.resize_bilinear(feat_out32, out_shape=(H, W))
        return feat_out, feat_out16, feat_out32


if __name__ == "__main__":
    import pickle
    paddle.disable_static()
    net = BiSeNet(19)
    param, _ = paddle.load('./resnet.pdparams')
    net.set_dict(param)
    net.eval()
    #print(net.state_dict().keys())
    #np.random.seed(2)
    #x = np.random.randn(16,3,640,480).astype(np.float32)
    with open('./x.pickle', 'rb') as f:
        x = pickle.load(f)
    in_ten = paddle.to_tensor(x)
    out, out16, out32 = net(in_ten)
    print(out.numpy().sum())
    with open('./out.pickle', 'wb') as f:
        pickle.dump(out.numpy(), f)
    print(out.shape)
    print(out16.shape)
    print(out32.shape)
