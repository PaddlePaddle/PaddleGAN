# code was heavily based on https://github.com/clovaai/stargan-v2
# Users should be careful about adopting these functions in any commercial matters.
# https://github.com/clovaai/stargan-v2#license
import paddle
from paddle import nn
import paddle.nn.functional as F

from .builder import GENERATORS
import numpy as np
import math

from ppgan.modules.wing import CoordConvTh, ConvBlock, HourGlass, preprocess

from ppgan.utils.download import get_path_from_url

FAN_WEIGHT_URL = "https://paddlegan.bj.bcebos.com/models/wing.pdparams"


class ResBlk(nn.Layer):
    def __init__(self,
                 dim_in,
                 dim_out,
                 actv=nn.LeakyReLU(0.2),
                 normalize=False,
                 downsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)
        self.maxpool = nn.AvgPool2D(kernel_size=2)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2D(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2D(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2D(dim_in,
                                           weight_attr=True,
                                           bias_attr=True)
            self.norm2 = nn.InstanceNorm2D(dim_in,
                                           weight_attr=True,
                                           bias_attr=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2D(dim_in, dim_out, 1, 1, 0, bias_attr=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = self.maxpool(x)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = self.maxpool(x)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance


class AdaIN(nn.Layer):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2D(num_features,
                                      weight_attr=False,
                                      bias_attr=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        h = self.fc(s)
        # h = h.view(h.size(0), h.size(1), 1, 1)
        h = paddle.reshape(h, (h.shape[0], h.shape[1], 1, 1))
        gamma, beta = paddle.chunk(h, chunks=2, axis=1)
        return (1 + gamma) * self.norm(x) + beta


class AdainResBlk(nn.Layer):
    def __init__(self,
                 dim_in,
                 dim_out,
                 style_dim=64,
                 w_hpf=0,
                 actv=nn.LeakyReLU(0.2),
                 upsample=False):
        super().__init__()
        self.w_hpf = w_hpf
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = nn.Conv2D(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2D(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2D(dim_in, dim_out, 1, 1, 0, bias_attr=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        if self.w_hpf == 0:
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


class HighPass(nn.Layer):
    def __init__(self, w_hpf):
        super(HighPass, self).__init__()
        self.filter = paddle.to_tensor([[-1, -1, -1], [-1, 8., -1],
                                        [-1, -1, -1]]) / w_hpf

    def forward(self, x):
        # filter = self.filter.unsqueeze(0).unsqueeze(1).repeat(x.size(1), 1, 1, 1)
        filter = self.filter.unsqueeze(0).unsqueeze(1).tile(
            [x.shape[1], 1, 1, 1])
        return F.conv2d(x, filter, padding=1, groups=x.shape[1])


@GENERATORS.register()
class StarGANv2Generator(nn.Layer):
    def __init__(self, img_size=256, style_dim=64, max_conv_dim=512, w_hpf=1):
        super().__init__()
        dim_in = 2**14 // img_size
        self.img_size = img_size
        self.from_rgb = nn.Conv2D(3, dim_in, 3, 1, 1)
        self.encode = nn.LayerList()
        self.decode = nn.LayerList()
        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2D(dim_in, weight_attr=True, bias_attr=True),
            nn.LeakyReLU(0.2), nn.Conv2D(dim_in, 3, 1, 1, 0))

        # down/up-sampling blocks
        repeat_num = int(np.log2(img_size)) - 4
        if w_hpf > 0:
            repeat_num += 1
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            self.encode.append(
                ResBlk(dim_in, dim_out, normalize=True, downsample=True))
            if len(self.decode) == 0:
                self.decode.append(
                    AdainResBlk(dim_out,
                                dim_in,
                                style_dim,
                                w_hpf=w_hpf,
                                upsample=True))
            else:
                self.decode.insert(0,
                                   AdainResBlk(dim_out,
                                               dim_in,
                                               style_dim,
                                               w_hpf=w_hpf,
                                               upsample=True))  # stack-like
            dim_in = dim_out

        # bottleneck blocks
        for _ in range(2):
            self.encode.append(ResBlk(dim_out, dim_out, normalize=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_out, style_dim, w_hpf=w_hpf))

        if w_hpf > 0:
            self.hpf = HighPass(w_hpf)

    def forward(self, x, s, masks=None):
        x = self.from_rgb(x)
        cache = {}
        for block in self.encode:
            if (masks is not None) and (x.shape[2] in [32, 64, 128]):
                cache[x.shape[2]] = x
            x = block(x)
        for block in self.decode:
            x = block(x, s)
            if (masks is not None) and (x.shape[2] in [32, 64, 128]):
                mask = masks[0] if x.shape[2] in [32] else masks[1]
                mask = F.interpolate(mask,
                                     size=[x.shape[2], x.shape[2]],
                                     mode='bilinear')
                x = x + self.hpf(mask * cache[x.shape[2]])
        return self.to_rgb(x)


@GENERATORS.register()
class StarGANv2Mapping(nn.Layer):
    def __init__(self, latent_dim=16, style_dim=64, num_domains=2):
        super().__init__()
        layers = []
        layers += [nn.Linear(latent_dim, 512)]
        layers += [nn.ReLU()]
        for _ in range(3):
            layers += [nn.Linear(512, 512)]
            layers += [nn.ReLU()]
        self.shared = nn.Sequential(*layers)

        self.unshared = nn.LayerList()
        for _ in range(num_domains):
            self.unshared.append(
                nn.Sequential(nn.Linear(512, 512),
                              nn.ReLU(), nn.Linear(512, 512), nn.ReLU(),
                              nn.Linear(512, 512), nn.ReLU(),
                              nn.Linear(512, style_dim)))

    def forward(self, z, y):
        h = self.shared(z)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = paddle.stack(out, axis=1)  # (batch, num_domains, style_dim)
        idx = paddle.to_tensor(np.array(range(y.shape[0]))).astype('int')
        s = []
        for i in range(idx.shape[0]):
            s += [
                out[idx[i].numpy().astype(np.int).tolist()[0],
                    y[i].numpy().astype(np.int).tolist()[0]]
            ]
        s = paddle.stack(s)
        s = paddle.reshape(s, (s.shape[0], -1))
        return s


@GENERATORS.register()
class StarGANv2Style(nn.Layer):
    def __init__(self,
                 img_size=256,
                 style_dim=64,
                 num_domains=2,
                 max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2D(3, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2D(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)

        self.unshared = nn.LayerList()
        for _ in range(num_domains):
            self.unshared.append(nn.Linear(dim_out, style_dim))

    def forward(self, x, y):
        h = self.shared(x)
        h = paddle.reshape(h, (h.shape[0], -1))
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = paddle.stack(out, axis=1)  # (batch, num_domains, style_dim)
        idx = paddle.to_tensor(np.array(range(y.shape[0]))).astype('int')
        s = []
        for i in range(idx.shape[0]):
            s += [
                out[idx[i].numpy().astype(np.int).tolist()[0],
                    y[i].numpy().astype(np.int).tolist()[0]]
            ]
        s = paddle.stack(s)
        s = paddle.reshape(s, (s.shape[0], -1))
        return s


@GENERATORS.register()
class FAN(nn.Layer):
    def __init__(self,
                 num_modules=1,
                 end_relu=False,
                 num_landmarks=98,
                 fname_pretrained=None):
        super(FAN, self).__init__()
        self.num_modules = num_modules
        self.end_relu = end_relu

        # Base part
        self.conv1 = CoordConvTh(256,
                                 256,
                                 True,
                                 False,
                                 in_channels=3,
                                 out_channels=64,
                                 kernel_size=7,
                                 stride=2,
                                 padding=3)
        self.bn1 = nn.BatchNorm2D(64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 128)
        self.conv4 = ConvBlock(128, 256)

        # Stacking part
        self.add_sublayer('m0', HourGlass(1, 4, 256, first_one=True))
        self.add_sublayer('top_m_0', ConvBlock(256, 256))
        self.add_sublayer('conv_last0', nn.Conv2D(256, 256, 1, 1, 0))
        self.add_sublayer('bn_end0', nn.BatchNorm2D(256))
        self.add_sublayer('l0', nn.Conv2D(256, num_landmarks + 1, 1, 1, 0))

        if fname_pretrained is not None:
            self.load_pretrained_weights(fname_pretrained)
        else:
            weight_path = get_path_from_url(FAN_WEIGHT_URL)
            self.load_pretrained_weights(weight_path)

    def load_pretrained_weights(self, fname):
        import pickle
        import six

        with open(fname, 'rb') as f:
            checkpoint = pickle.load(f) if six.PY2 else pickle.load(
                f, encoding='latin1')

        model_weights = self.state_dict()
        model_weights.update({
            k: v
            for k, v in checkpoint['state_dict'].items() if k in model_weights
        })
        self.set_state_dict(model_weights)

    def forward(self, x):
        x, _ = self.conv1(x)
        x = F.relu(self.bn1(x), True)
        x = F.avg_pool2d(self.conv2(x), 2, stride=2)
        x = self.conv3(x)
        x = self.conv4(x)

        outputs = []
        boundary_channels = []
        tmp_out = None
        ll, boundary_channel = self._sub_layers['m0'](x, tmp_out)
        ll = self._sub_layers['top_m_0'](ll)
        ll = F.relu(
            self._sub_layers['bn_end0'](self._sub_layers['conv_last0'](ll)),
            True)

        # Predict heatmaps
        tmp_out = self._sub_layers['l0'](ll)
        if self.end_relu:
            tmp_out = F.relu(tmp_out)  # HACK: Added relu
        outputs.append(tmp_out)
        boundary_channels.append(boundary_channel)
        return outputs, boundary_channels

    @paddle.no_grad()
    def get_heatmap(self, x, b_preprocess=True):
        ''' outputs 0-1 normalized heatmap '''
        x = F.interpolate(x, size=[256, 256], mode='bilinear')
        x_01 = x * 0.5 + 0.5
        outputs, _ = self(x_01)
        heatmaps = outputs[-1][:, :-1, :, :]
        scale_factor = x.shape[2] // heatmaps.shape[2]
        if b_preprocess:
            heatmaps = F.interpolate(heatmaps,
                                     scale_factor=scale_factor,
                                     mode='bilinear',
                                     align_corners=True)
            heatmaps = preprocess(heatmaps)
        return heatmaps
