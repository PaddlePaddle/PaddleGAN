import paddle
import paddle.nn as nn
import numpy as np
from ppgan.utils.download import get_path_from_url

model_urls = {
    'caffevgg19': ('https://paddlegan.bj.bcebos.com/models/vgg19_no_fc.npy',
                   '8ea1ef2374f8684b6cea9f300849be81')
}


class CaffeVGG19(nn.Layer):
    cfg = [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512,
        'M', 512, 512, 512, 512, 'M'
    ]

    def __init__(self, output_index: int = 26) -> None:
        super().__init__()
        arch = 'caffevgg19'
        weights_path = get_path_from_url(model_urls[arch][0],
                                         model_urls[arch][1])
        data_dict: dict = np.load(weights_path,
                                  encoding='latin1',
                                  allow_pickle=True).item()
        self.features = self.make_layers(self.cfg, data_dict)
        del data_dict
        self.features = nn.Sequential(*self.features.sublayers()[:output_index])
        mean = paddle.to_tensor([103.939, 116.779, 123.68])
        self.mean = mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    def _process(self, x):
        # value to 255
        rgb = (x * 0.5 + 0.5) * 255
        # rgb to bgr
        bgr = paddle.stack((rgb[:, 2, :, :], rgb[:, 1, :, :], rgb[:, 0, :, :]),
                           1)
        # vgg norm
        return bgr - self.mean

    def _forward_impl(self, x):
        x = self._process(x)
        # NOTE get output with out relu activation
        x = self.features(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)

    @staticmethod
    def get_conv_filter(data_dict, name):
        return data_dict[name][0]

    @staticmethod
    def get_bias(data_dict, name):
        return data_dict[name][1]

    @staticmethod
    def get_fc_weight(data_dict, name):
        return data_dict[name][0]

    def make_layers(self, cfg, data_dict, batch_norm=False) -> nn.Sequential:
        layers = []
        in_channels = 3
        block = 1
        number = 1
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2D(kernel_size=2, stride=2)]
                block += 1
                number = 1
            else:
                conv2d = nn.Conv2D(in_channels, v, kernel_size=3, padding=1)
                """ set value """
                weight = paddle.to_tensor(
                    self.get_conv_filter(data_dict, f'conv{block}_{number}'))
                weight = weight.transpose((3, 2, 0, 1))
                bias = paddle.to_tensor(
                    self.get_bias(data_dict, f'conv{block}_{number}'))
                conv2d.weight.set_value(weight)
                conv2d.bias.set_value(bias)
                number += 1
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2D(v), nn.ReLU()]
                else:
                    layers += [conv2d, nn.ReLU()]
                in_channels = v

        return nn.Sequential(*layers)
