import paddle
from paddle import nn
from paddle.nn import functional as F


class ConvBNRelu(nn.Layer):
    def __init__(self,
                 cin,
                 cout,
                 kernel_size,
                 stride,
                 padding,
                 residual=False,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
            nn.Conv2D(cin, cout, kernel_size, stride, padding),
            nn.BatchNorm2D(cout))
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


class NonNormConv2d(nn.Layer):
    def __init__(self,
                 cin,
                 cout,
                 kernel_size,
                 stride,
                 padding,
                 residual=False,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
            nn.Conv2D(cin, cout, kernel_size, stride, padding), )
        self.act = nn.LeakyReLU(0.01)

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)


class Conv2dTransposeRelu(nn.Layer):
    def __init__(self,
                 cin,
                 cout,
                 kernel_size,
                 stride,
                 padding,
                 output_padding=0,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
            nn.Conv2DTranspose(cin, cout, kernel_size, stride, padding,
                               output_padding), nn.BatchNorm2D(cout))
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)
