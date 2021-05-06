# Day 3 作业--Pixel2Pixel：人像卡通化

经过今天的学习，相信大家对图像翻译、风格迁移有了一定的了解啦，是不是也想自己动手来实现下呢？

那么，为了满足大家动手实践的愿望，同时为了巩固大家学到的知识，我们Day 3的作业便是带大家完成一遍课程讲解过的应用--**Pixel2Pixel：人像卡通化**

在本次作业中，大家需要做的是：**补齐代码，跑通训练，提交一张卡通化的成品图，动手完成自己的第一个人像卡通化的应用~**

![](https://ai-studio-static-online.cdn.bcebos.com/6e3af14bf9f847ab92215753fb3b8f61a66186b538f44da78ca56627c35717b8)

## 准备工作：引入依赖 & 数据准备


```python
import paddle
import paddle.nn as nn
from paddle.io import Dataset, DataLoader

import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

%matplotlib inline
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/__init__.py:107: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import MutableMapping
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/rcsetup.py:20: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import Iterable, Mapping
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/colors.py:53: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import Sized


### 数据准备：

- 真人数据来自[seeprettyface](http://www.seeprettyface.com/mydataset.html)。
- 数据预处理（详情见[photo2cartoon](https://github.com/minivision-ai/photo2cartoon)项目）。
<div>
  <img src='https://ai-studio-static-online.cdn.bcebos.com/c56c889827534363a8b6909d7737a1da64635ad33e1e44cb822f4c1cf1dfc689' height='1000px' width='1000px'>
</div>

- 使用[photo2cartoon](https://github.com/minivision-ai/photo2cartoon)项目生成真人数据对应的卡通数据。


```python
# 解压数据
!unzip -qao data/data79149/cartoon_A2B.zip -d data/
```

### 数据可视化


```python
# 训练数据统计
train_names = os.listdir('data/cartoon_A2B/train')
print(f'训练集数据量: {len(train_names)}')

# 测试数据统计
test_names = os.listdir('data/cartoon_A2B/test')
print(f'测试集数据量: {len(test_names)}')

# 训练数据可视化
imgs = []
for img_name in np.random.choice(train_names, 3, replace=False):
    imgs.append(cv2.imread('data/cartoon_A2B/train/'+img_name))

img_show = np.vstack(imgs)[:,:,::-1]
plt.figure(figsize=(10, 10))
plt.imshow(img_show)
plt.show()
```

    训练集数据量: 1361
    测试集数据量: 100


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/cbook/__init__.py:2349: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      if isinstance(obj, collections.Iterator):
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/cbook/__init__.py:2366: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      return list(data) if isinstance(data, collections.MappingView) else data



![png](output_6_2.png)



```python
class PairedData(Dataset):
    def __init__(self, phase):
        super(PairedData, self).__init__() 
        self.img_path_list = self.load_A2B_data(phase)    # 获取数据列表
        self.num_samples = len(self.img_path_list)        # 数据量

    def __getitem__(self, idx):
        img_A2B = cv2.imread(self.img_path_list[idx])     # 读取一组数据
        img_A2B = img_A2B.astype('float32') / 127.5 - 1.  # 从0~255归一化至-1~1
        img_A2B = img_A2B.transpose(2, 0, 1)              # 维度变换HWC -> CHW
        img_A = img_A2B[..., :256]                        # 真人照
        img_B = img_A2B[..., 256:]                        # 卡通图
        return img_A, img_B

    def __len__(self):
        return self.num_samples

    @staticmethod
    def load_A2B_data(phase):
        assert phase in ['train', 'test'], "phase should be set within ['train', 'test']"
        # 读取数据集，数据中每张图像包含照片和对应的卡通画。
        data_path = 'data/cartoon_A2B/'+phase
        return [os.path.join(data_path, x) for x in os.listdir(data_path)]
```


```python
paired_dataset_train = PairedData('train')
paired_dataset_test = PairedData('test')
```

## 第一步：搭建生成器

### 请大家补齐空白处的代码，‘#’ 后是提示。


```python
class UnetGenerator(nn.Layer):
    def __init__(self, input_nc=3, output_nc=3, ngf=64):
        super(UnetGenerator, self).__init__()

        self.down1 = nn.Conv2D(input_nc, ngf, kernel_size=4, stride=2, padding=1)
        self.down2 = Downsample(ngf, ngf*2)
        self.down3 = Downsample(ngf*2, ngf*4)
        self.down4 = Downsample(ngf*4, ngf*8)
        self.down5 = Downsample(ngf*8, ngf*8)
        self.down6 = Downsample(ngf*8, ngf*8)
        self.down7 = Downsample(ngf*8, ngf*8)

        self.center = Downsample(ngf*8, ngf*8)

        self.up7 = Upsample(ngf*8, ngf*8, use_dropout=True)
        self.up6 = Upsample(ngf*8*2, ngf*8, use_dropout=True)
        self.up5 = Upsample(ngf*8*2, ngf*8, use_dropout=True)
        self.up4 = Upsample(ngf*8*2, ngf*8)
        self.up3 = Upsample(ngf*8*2, ngf*4)
        self.up2 = Upsample(ngf*4*2, ngf*2)
        self.up1 = Upsample(ngf*2*2, ngf)

        self.output_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2DTranspose(ngf*2, output_nc, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        
        c = self.center(d7)
        
        x = self.up7(c, d7)
        x = self.up6(x, d6)
        x = self.up5(x, d5)
        x = self.up4(x, d4)
        x = self.up3(x, d3)
        x = self.up2(x, d2)
        x = self.up1(x, d1)

        x = self.output_block(x)
        return x


class Downsample(nn.Layer):
    # LeakyReLU => conv => batch norm
    def __init__(self, in_dim, out_dim, kernel_size=4, stride=2, padding=1):
        super(Downsample, self).__init__()

        self.layers = nn.Sequential(
            nn.LeakyReLU(0.2),                                                                 # LeakyReLU, leaky=0.2
            nn.Conv2D(in_dim, out_dim, kernel_size, stride, padding, bias_attr=False),         # Conv2D
            nn.BatchNorm2D(out_dim)        
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class Upsample(nn.Layer):
    # ReLU => deconv => batch norm => dropout
    def __init__(self, in_dim, out_dim, kernel_size=4, stride=2, padding=1, use_dropout=False):
        super(Upsample, self).__init__()

        sequence = [
            nn.ReLU(),                                                                          # ReLU
            nn.Conv2DTranspose(in_dim, out_dim, kernel_size, stride, padding, bias_attr=False), # Conv2DTranspose
            nn.BatchNorm2D(out_dim)   
        ]

        if use_dropout:
            sequence.append(nn.Dropout(p=0.5))

        self.layers = nn.Sequential(*sequence)

    def forward(self, x, skip):
        x = self.layers(x)
        x = paddle.concat([x, skip], axis=1)
        return x
```

## 第二步：鉴别器的搭建

### 请大家补齐空白处的代码，‘#’ 后是提示。


```python
class NLayerDiscriminator(nn.Layer):
    def __init__(self, input_nc=6, ndf=64):
        super(NLayerDiscriminator, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2D(input_nc, ndf, kernel_size=4, stride=2, padding=1), 
            nn.LeakyReLU(0.2),
            
            ConvBlock(ndf, ndf*2),
            ConvBlock(ndf*2, ndf*4),
            ConvBlock(ndf*4, ndf*8, stride=1),

            nn.Conv2D(ndf*8, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.layers(input)


class ConvBlock(nn.Layer):
    # conv => batch norm => LeakyReLU
    def __init__(self, in_dim, out_dim, kernel_size=4, stride=2, padding=1):
        super(ConvBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2D(in_dim, out_dim, kernel_size, stride, padding, bias_attr=False),  # Conv2D
            nn.BatchNorm2D(out_dim),                                                    # BatchNorm2D
            nn.LeakyReLU(0.2)     
        )

    def forward(self, x):
        x = self.layers(x)
        return x
```


```python
generator = UnetGenerator()
discriminator = NLayerDiscriminator()
```


```python
out = generator(paddle.ones([1, 3, 256, 256]))
print('生成器输出尺寸：', out.shape)  # 应为[1, 3, 256, 256]

out = discriminator(paddle.ones([1, 6, 256, 256]))
print('鉴别器输出尺寸：', out.shape)  # 应为[1, 1, 30, 30]
```

    生成器输出尺寸： [1, 3, 256, 256]
    鉴别器输出尺寸： [1, 1, 30, 30]


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/nn/layer/norm.py:648: UserWarning: When training, we now always track global mean and variance.
      "When training, we now always track global mean and variance.")



```python
# 超参数
LR = 1e-4
BATCH_SIZE = 8
EPOCHS = 100

# 优化器
optimizerG = paddle.optimizer.Adam(
    learning_rate=LR,
    parameters=generator.parameters(),
    beta1=0.5,
    beta2=0.999)

optimizerD = paddle.optimizer.Adam(
    learning_rate=LR,
    parameters=discriminator.parameters(), 
    beta1=0.5,
    beta2=0.999)
    
# 损失函数
bce_loss = nn.BCELoss()
l1_loss = nn.L1Loss()


# dataloader
data_loader_train = DataLoader(
    paired_dataset_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
    )

data_loader_test = DataLoader(
    paired_dataset_test,
    batch_size=BATCH_SIZE
    )
```


```python
results_save_path = 'work/results'
os.makedirs(results_save_path, exist_ok=True)  # 保存每个epoch的测试结果

weights_save_path = 'work/weights'
os.makedirs(weights_save_path, exist_ok=True)  # 保存模型

for epoch in range(EPOCHS):
    for data in tqdm(data_loader_train):
        real_A, real_B = data
        
        optimizerD.clear_grad()
        # D([real_A, real_B])
        real_AB = paddle.concat((real_A, real_B), 1)
        d_real_predict = discriminator(real_AB)
        d_real_loss = bce_loss(d_real_predict, paddle.ones_like(d_real_predict))

        # D([real_A, fake_B])
        fake_B = generator(real_A).detach()
        fake_AB = paddle.concat((real_A, fake_B), 1)
        d_fake_predict = discriminator(fake_AB)
        d_fake_loss = bce_loss(d_fake_predict, paddle.zeros_like(d_fake_predict))
        
        # train D
        d_loss = (d_real_loss + d_fake_loss) / 2.
        d_loss.backward()
        optimizerD.step()

        optimizerG.clear_grad()
        # D([real_A, fake_B])
        fake_B = generator(real_A)
        fake_AB = paddle.concat((real_A, fake_B), 1)
        g_fake_predict = discriminator(fake_AB)
        g_bce_loss = bce_loss(g_fake_predict, paddle.ones_like(g_fake_predict))
        g_l1_loss = l1_loss(fake_B, real_B) * 100.
        g_loss = g_bce_loss + g_l1_loss * 1.   
        
        # train G
        g_loss.backward()
        optimizerG.step()

    print(f'Epoch [{epoch+1}/{EPOCHS}] Loss D: {d_loss.numpy()}, Loss G: {g_loss.numpy()}')

    if (epoch+1) % 10 == 0:
        paddle.save(generator.state_dict(), os.path.join(weights_save_path, 'epoch'+str(epoch+1).zfill(3)+'.pdparams'))

        # test
        generator.eval()
        with paddle.no_grad():
            for data in data_loader_test:
                real_A, real_B = data
                break

            fake_B = generator(real_A)
            result = paddle.concat([real_A[:3], real_B[:3], fake_B[:3]], 3)

            result = result.detach().numpy().transpose(0, 2, 3, 1)
            result = np.vstack(result)
            result = (result * 127.5 + 127.5).astype(np.uint8)
    
        cv2.imwrite(os.path.join(results_save_path, 'epoch'+str(epoch+1).zfill(3)+'.png'), result)

        generator.train()
```

    100%|██████████| 170/170 [00:23<00:00,  7.22it/s]
      1%|          | 1/170 [00:00<00:24,  6.87it/s]

    Epoch [1/100] Loss D: [0.5284234], Loss G: [13.064737]


    100%|██████████| 170/170 [00:23<00:00,  7.29it/s]
      1%|          | 1/170 [00:00<00:25,  6.64it/s]

    Epoch [2/100] Loss D: [0.54123324], Loss G: [11.698286]


    100%|██████████| 170/170 [00:23<00:00,  7.29it/s]
      1%|          | 1/170 [00:00<00:24,  6.78it/s]

    Epoch [3/100] Loss D: [0.5299478], Loss G: [11.66753]


    100%|██████████| 170/170 [00:23<00:00,  7.28it/s]
      1%|          | 1/170 [00:00<00:24,  6.85it/s]

    Epoch [4/100] Loss D: [0.6474249], Loss G: [11.17137]


      1%|          | 2/170 [00:00<00:24,  6.96it/s]100%|██████████| 170/170 [00:23<00:00,  7.27it/s]
      1%|          | 1/170 [00:00<00:24,  6.86it/s]

    Epoch [5/100] Loss D: [0.6039876], Loss G: [11.128557]


      1%|          | 2/170 [00:00<00:24,  6.97it/s]100%|██████████| 170/170 [00:23<00:00,  7.28it/s]
      1%|          | 1/170 [00:00<00:24,  6.80it/s]

    Epoch [6/100] Loss D: [0.5950774], Loss G: [10.30802]


      1%|          | 2/170 [00:00<00:24,  6.91it/s]100%|██████████| 170/170 [00:23<00:00,  7.27it/s]
      1%|          | 1/170 [00:00<00:24,  6.78it/s]

    Epoch [7/100] Loss D: [0.6796185], Loss G: [10.197595]


      1%|          | 2/170 [00:00<00:24,  6.90it/s]100%|██████████| 170/170 [00:23<00:00,  7.24it/s]
      1%|          | 1/170 [00:00<00:24,  6.78it/s]

    Epoch [8/100] Loss D: [0.6318821], Loss G: [10.000329]


      1%|          | 2/170 [00:00<00:24,  6.91it/s]100%|██████████| 170/170 [00:23<00:00,  7.27it/s]
      1%|          | 1/170 [00:00<00:25,  6.72it/s]

    Epoch [9/100] Loss D: [0.5950939], Loss G: [9.717655]


    100%|██████████| 170/170 [00:23<00:00,  7.24it/s]


    Epoch [10/100] Loss D: [0.5593741], Loss G: [9.465918]


    100%|██████████| 170/170 [00:23<00:00,  7.26it/s]
      1%|          | 1/170 [00:00<00:24,  6.87it/s]

    Epoch [11/100] Loss D: [0.54420507], Loss G: [8.719758]


    100%|██████████| 170/170 [00:23<00:00,  7.27it/s]
      1%|          | 1/170 [00:00<00:25,  6.73it/s]

    Epoch [12/100] Loss D: [0.6362987], Loss G: [8.977068]


    100%|██████████| 170/170 [00:23<00:00,  7.27it/s]
      1%|          | 1/170 [00:00<00:24,  6.88it/s]

    Epoch [13/100] Loss D: [0.6640139], Loss G: [9.174168]


    100%|██████████| 170/170 [00:23<00:00,  7.30it/s]
      1%|          | 1/170 [00:00<00:25,  6.73it/s]

    Epoch [14/100] Loss D: [0.6028167], Loss G: [9.190138]


    100%|██████████| 170/170 [00:23<00:00,  7.30it/s]
      1%|          | 1/170 [00:00<00:24,  6.90it/s]

    Epoch [15/100] Loss D: [0.67779934], Loss G: [8.241856]


    100%|██████████| 170/170 [00:23<00:00,  7.30it/s]
      1%|          | 1/170 [00:00<00:24,  6.82it/s]

    Epoch [16/100] Loss D: [0.69265723], Loss G: [8.520813]


    100%|██████████| 170/170 [00:23<00:00,  7.27it/s]
      1%|          | 1/170 [00:00<00:25,  6.74it/s]

    Epoch [17/100] Loss D: [0.5550152], Loss G: [8.240015]


      1%|          | 2/170 [00:00<00:24,  6.88it/s]100%|██████████| 170/170 [00:23<00:00,  7.24it/s]
      1%|          | 1/170 [00:00<00:24,  6.83it/s]

    Epoch [18/100] Loss D: [0.7933258], Loss G: [7.9467487]


      1%|          | 2/170 [00:00<00:24,  6.95it/s]100%|██████████| 170/170 [00:23<00:00,  7.29it/s]
      1%|          | 1/170 [00:00<00:24,  6.86it/s]

    Epoch [19/100] Loss D: [0.6194705], Loss G: [8.075335]


      1%|          | 2/170 [00:00<00:24,  6.98it/s]100%|██████████| 170/170 [00:23<00:00,  7.28it/s]


    Epoch [20/100] Loss D: [0.66503227], Loss G: [7.8089933]


    100%|██████████| 170/170 [00:23<00:00,  7.29it/s]
      1%|          | 1/170 [00:00<00:24,  6.85it/s]

    Epoch [21/100] Loss D: [0.67030644], Loss G: [7.367738]


      1%|          | 2/170 [00:00<00:24,  6.99it/s]100%|██████████| 170/170 [00:23<00:00,  7.30it/s]
      1%|          | 1/170 [00:00<00:24,  6.80it/s]

    Epoch [22/100] Loss D: [0.631791], Loss G: [7.708729]


    100%|██████████| 170/170 [00:23<00:00,  7.29it/s]
      1%|          | 1/170 [00:00<00:25,  6.55it/s]

    Epoch [23/100] Loss D: [0.6423906], Loss G: [7.2092514]


      1%|          | 2/170 [00:00<00:24,  6.73it/s]100%|██████████| 170/170 [00:23<00:00,  7.30it/s]
      1%|          | 1/170 [00:00<00:24,  6.90it/s]

    Epoch [24/100] Loss D: [0.6739343], Loss G: [7.237084]


    100%|██████████| 170/170 [00:23<00:00,  7.30it/s]
      1%|          | 1/170 [00:00<00:24,  6.88it/s]

    Epoch [25/100] Loss D: [0.66729414], Loss G: [6.767889]


    100%|██████████| 170/170 [00:23<00:00,  7.29it/s]
      1%|          | 1/170 [00:00<00:24,  6.82it/s]

    Epoch [26/100] Loss D: [0.61588544], Loss G: [6.787832]


    100%|██████████| 170/170 [00:23<00:00,  7.30it/s]
      1%|          | 1/170 [00:00<00:24,  6.89it/s]

    Epoch [27/100] Loss D: [0.6792059], Loss G: [6.9433084]


    100%|██████████| 170/170 [00:23<00:00,  7.30it/s]
      1%|          | 1/170 [00:00<00:24,  6.88it/s]

    Epoch [28/100] Loss D: [0.72314185], Loss G: [6.8468137]


    100%|██████████| 170/170 [00:23<00:00,  7.28it/s]
      1%|          | 1/170 [00:00<00:24,  6.91it/s]

    Epoch [29/100] Loss D: [0.65756834], Loss G: [6.8010254]


      1%|          | 2/170 [00:00<00:23,  7.01it/s]100%|██████████| 170/170 [00:23<00:00,  7.29it/s]


    Epoch [30/100] Loss D: [0.6856936], Loss G: [6.6264324]


    100%|██████████| 170/170 [00:23<00:00,  7.29it/s]
      1%|          | 1/170 [00:00<00:24,  6.86it/s]

    Epoch [31/100] Loss D: [0.71327627], Loss G: [6.605694]


    100%|██████████| 170/170 [00:23<00:00,  7.31it/s]
      1%|          | 1/170 [00:00<00:25,  6.76it/s]

    Epoch [32/100] Loss D: [0.6019069], Loss G: [6.340989]


    100%|██████████| 170/170 [00:23<00:00,  7.30it/s]
      1%|          | 1/170 [00:00<00:24,  6.85it/s]

    Epoch [33/100] Loss D: [0.6134093], Loss G: [6.2769694]


      1%|          | 2/170 [00:00<00:24,  6.94it/s]100%|██████████| 170/170 [00:23<00:00,  7.30it/s]
      1%|          | 1/170 [00:00<00:24,  6.87it/s]

    Epoch [34/100] Loss D: [0.67665935], Loss G: [6.7924075]


      1%|          | 2/170 [00:00<00:24,  6.98it/s]100%|██████████| 170/170 [00:23<00:00,  7.30it/s]
      1%|          | 1/170 [00:00<00:24,  6.83it/s]

    Epoch [35/100] Loss D: [0.6352638], Loss G: [6.151738]


      1%|          | 2/170 [00:00<00:24,  6.96it/s]100%|██████████| 170/170 [00:23<00:00,  7.30it/s]
      1%|          | 1/170 [00:00<00:24,  6.91it/s]

    Epoch [36/100] Loss D: [0.7007777], Loss G: [6.3575606]


      1%|          | 2/170 [00:00<00:23,  7.02it/s]100%|██████████| 170/170 [00:23<00:00,  7.30it/s]
      1%|          | 1/170 [00:00<00:24,  6.83it/s]

    Epoch [37/100] Loss D: [0.69342726], Loss G: [5.850954]


    100%|██████████| 170/170 [00:23<00:00,  7.29it/s]
      1%|          | 1/170 [00:00<00:25,  6.54it/s]

    Epoch [38/100] Loss D: [0.7059885], Loss G: [6.045345]


      1%|          | 2/170 [00:00<00:24,  6.73it/s]100%|██████████| 170/170 [00:23<00:00,  7.29it/s]
      1%|          | 1/170 [00:00<00:25,  6.71it/s]

    Epoch [39/100] Loss D: [0.5827794], Loss G: [5.97258]


      1%|          | 2/170 [00:00<00:24,  6.86it/s]100%|██████████| 170/170 [00:23<00:00,  7.30it/s]


    Epoch [40/100] Loss D: [0.6818665], Loss G: [5.9698906]


    100%|██████████| 170/170 [00:23<00:00,  7.28it/s]
      1%|          | 1/170 [00:00<00:25,  6.54it/s]

    Epoch [41/100] Loss D: [0.6436964], Loss G: [5.5804996]


    100%|██████████| 170/170 [00:23<00:00,  7.30it/s]
      1%|          | 1/170 [00:00<00:24,  6.78it/s]

    Epoch [42/100] Loss D: [0.6325854], Loss G: [5.329829]


    100%|██████████| 170/170 [00:23<00:00,  7.30it/s]
      1%|          | 1/170 [00:00<00:24,  6.82it/s]

    Epoch [43/100] Loss D: [0.6460885], Loss G: [5.8121433]


    100%|██████████| 170/170 [00:23<00:00,  7.30it/s]
      1%|          | 1/170 [00:00<00:24,  6.91it/s]

    Epoch [44/100] Loss D: [0.5965668], Loss G: [5.508972]


    100%|██████████| 170/170 [00:23<00:00,  7.22it/s]
      1%|          | 1/170 [00:00<00:24,  6.78it/s]

    Epoch [45/100] Loss D: [0.84220695], Loss G: [5.683806]


    100%|██████████| 170/170 [00:23<00:00,  7.31it/s]
      1%|          | 1/170 [00:00<00:24,  6.81it/s]

    Epoch [46/100] Loss D: [0.63925695], Loss G: [5.52334]


    100%|██████████| 170/170 [00:23<00:00,  7.30it/s]
      1%|          | 1/170 [00:00<00:24,  6.82it/s]

    Epoch [47/100] Loss D: [0.6487623], Loss G: [5.21964]


    100%|██████████| 170/170 [00:23<00:00,  7.30it/s]
      1%|          | 1/170 [00:00<00:24,  6.80it/s]

    Epoch [48/100] Loss D: [0.6693777], Loss G: [5.29474]


    100%|██████████| 170/170 [00:23<00:00,  7.30it/s]
      1%|          | 1/170 [00:00<00:24,  6.88it/s]

    Epoch [49/100] Loss D: [0.6117195], Loss G: [5.296708]


    100%|██████████| 170/170 [00:23<00:00,  7.30it/s]


    Epoch [50/100] Loss D: [0.67985165], Loss G: [5.4502654]


    100%|██████████| 170/170 [00:23<00:00,  7.29it/s]
      1%|          | 1/170 [00:00<00:24,  6.93it/s]

    Epoch [51/100] Loss D: [0.64274585], Loss G: [4.8949633]


    100%|██████████| 170/170 [00:23<00:00,  7.31it/s]
      1%|          | 1/170 [00:00<00:24,  6.78it/s]

    Epoch [52/100] Loss D: [0.62808985], Loss G: [5.282054]


    100%|██████████| 170/170 [00:23<00:00,  7.30it/s]
      1%|          | 1/170 [00:00<00:27,  6.12it/s]

    Epoch [53/100] Loss D: [0.6447178], Loss G: [5.0051484]


      1%|          | 2/170 [00:00<00:26,  6.43it/s]100%|██████████| 170/170 [00:23<00:00,  7.30it/s]
      1%|          | 1/170 [00:00<00:24,  6.78it/s]

    Epoch [54/100] Loss D: [0.6399511], Loss G: [5.000868]


    100%|██████████| 170/170 [00:23<00:00,  7.30it/s]
      1%|          | 1/170 [00:00<00:24,  6.83it/s]

    Epoch [55/100] Loss D: [0.6100814], Loss G: [5.101789]


      1%|          | 2/170 [00:00<00:24,  6.95it/s]100%|██████████| 170/170 [00:23<00:00,  7.31it/s]
      1%|          | 1/170 [00:00<00:24,  6.82it/s]

    Epoch [56/100] Loss D: [0.6472188], Loss G: [5.447066]


    100%|██████████| 170/170 [00:23<00:00,  7.29it/s]
      1%|          | 1/170 [00:00<00:24,  6.86it/s]

    Epoch [57/100] Loss D: [0.6713679], Loss G: [5.348196]


    100%|██████████| 170/170 [00:23<00:00,  7.28it/s]
      1%|          | 1/170 [00:00<00:24,  6.86it/s]

    Epoch [58/100] Loss D: [0.615517], Loss G: [4.9242]


    100%|██████████| 170/170 [00:23<00:00,  7.30it/s]
      1%|          | 1/170 [00:00<00:24,  6.85it/s]

    Epoch [59/100] Loss D: [0.6416937], Loss G: [5.0159454]


      1%|          | 2/170 [00:00<00:24,  6.95it/s]100%|██████████| 170/170 [00:23<00:00,  7.30it/s]


    Epoch [60/100] Loss D: [0.675475], Loss G: [4.7302303]


    100%|██████████| 170/170 [00:23<00:00,  7.30it/s]
      1%|          | 1/170 [00:00<00:24,  6.85it/s]

    Epoch [61/100] Loss D: [0.6496334], Loss G: [4.6984844]


    100%|██████████| 170/170 [00:23<00:00,  7.31it/s]
      1%|          | 1/170 [00:00<00:25,  6.71it/s]

    Epoch [62/100] Loss D: [0.6449231], Loss G: [4.8061175]


    100%|██████████| 170/170 [00:23<00:00,  7.30it/s]
      1%|          | 1/170 [00:00<00:24,  6.77it/s]

    Epoch [63/100] Loss D: [0.66442376], Loss G: [5.032484]


      1%|          | 2/170 [00:00<00:24,  6.92it/s]100%|██████████| 170/170 [00:23<00:00,  7.30it/s]
      1%|          | 1/170 [00:00<00:24,  6.79it/s]

    Epoch [64/100] Loss D: [0.6816081], Loss G: [4.970877]


    100%|██████████| 170/170 [00:23<00:00,  7.30it/s]
      1%|          | 1/170 [00:00<00:24,  6.77it/s]

    Epoch [65/100] Loss D: [0.68846303], Loss G: [5.0442305]


    100%|██████████| 170/170 [00:23<00:00,  7.29it/s]
      1%|          | 1/170 [00:00<00:24,  6.76it/s]

    Epoch [66/100] Loss D: [0.6589049], Loss G: [4.9111423]


    100%|██████████| 170/170 [00:23<00:00,  7.30it/s]
      1%|          | 1/170 [00:00<00:24,  6.77it/s]

    Epoch [67/100] Loss D: [0.6767062], Loss G: [5.03377]


      1%|          | 2/170 [00:00<00:24,  6.91it/s]100%|██████████| 170/170 [00:23<00:00,  7.30it/s]
      1%|          | 1/170 [00:00<00:24,  6.80it/s]

    Epoch [68/100] Loss D: [0.6409887], Loss G: [4.5729055]


      1%|          | 2/170 [00:00<00:24,  6.94it/s]100%|██████████| 170/170 [00:23<00:00,  7.29it/s]
      1%|          | 1/170 [00:00<00:25,  6.71it/s]

    Epoch [69/100] Loss D: [0.658414], Loss G: [4.670569]


    100%|██████████| 170/170 [00:23<00:00,  7.30it/s]


    Epoch [70/100] Loss D: [0.6667415], Loss G: [4.7972655]


    100%|██████████| 170/170 [00:23<00:00,  7.27it/s]
      1%|          | 1/170 [00:00<00:25,  6.71it/s]

    Epoch [71/100] Loss D: [0.6540038], Loss G: [4.5085125]


      1%|          | 2/170 [00:00<00:24,  6.88it/s]100%|██████████| 170/170 [00:23<00:00,  7.29it/s]
      1%|          | 1/170 [00:00<00:25,  6.75it/s]

    Epoch [72/100] Loss D: [0.6422181], Loss G: [4.565335]


      1%|          | 2/170 [00:00<00:24,  6.91it/s]100%|██████████| 170/170 [00:23<00:00,  7.27it/s]
      1%|          | 1/170 [00:00<00:24,  6.83it/s]

    Epoch [73/100] Loss D: [0.669103], Loss G: [4.978608]


      1%|          | 2/170 [00:00<00:24,  6.90it/s]100%|██████████| 170/170 [00:23<00:00,  7.29it/s]
      1%|          | 1/170 [00:00<00:24,  6.82it/s]

    Epoch [74/100] Loss D: [0.6967082], Loss G: [4.2552986]


    100%|██████████| 170/170 [00:23<00:00,  7.30it/s]
      1%|          | 1/170 [00:00<00:24,  6.86it/s]

    Epoch [75/100] Loss D: [0.6473232], Loss G: [4.4088635]


    100%|██████████| 170/170 [00:23<00:00,  7.26it/s]
      1%|          | 1/170 [00:00<00:25,  6.76it/s]

    Epoch [76/100] Loss D: [0.70706826], Loss G: [4.3607]


    100%|██████████| 170/170 [00:23<00:00,  7.27it/s]
      1%|          | 1/170 [00:00<00:24,  6.82it/s]

    Epoch [77/100] Loss D: [0.6994731], Loss G: [4.878913]


    100%|██████████| 170/170 [00:23<00:00,  7.28it/s]
      1%|          | 1/170 [00:00<00:25,  6.62it/s]

    Epoch [78/100] Loss D: [0.67322636], Loss G: [4.7822013]


      1%|          | 2/170 [00:00<00:24,  6.80it/s]100%|██████████| 170/170 [00:23<00:00,  7.27it/s]
      1%|          | 1/170 [00:00<00:24,  6.88it/s]

    Epoch [79/100] Loss D: [0.6506158], Loss G: [4.8017626]


      2%|▏         | 3/170 [00:00<00:23,  7.10it/s]100%|██████████| 170/170 [00:23<00:00,  7.28it/s]


    Epoch [80/100] Loss D: [0.66648376], Loss G: [4.4588194]


    100%|██████████| 170/170 [00:23<00:00,  7.24it/s]
      1%|          | 1/170 [00:00<00:24,  6.89it/s]

    Epoch [81/100] Loss D: [0.6700024], Loss G: [4.300681]


    100%|██████████| 170/170 [00:23<00:00,  7.30it/s]
      1%|          | 1/170 [00:00<00:25,  6.63it/s]

    Epoch [82/100] Loss D: [0.6441313], Loss G: [4.6373615]


      1%|          | 2/170 [00:00<00:24,  6.80it/s]100%|██████████| 170/170 [00:23<00:00,  7.30it/s]
      1%|          | 1/170 [00:00<00:25,  6.70it/s]

    Epoch [83/100] Loss D: [0.66145974], Loss G: [4.4312263]


      1%|          | 2/170 [00:00<00:24,  6.86it/s]100%|██████████| 170/170 [00:23<00:00,  7.28it/s]
      1%|          | 1/170 [00:00<00:25,  6.70it/s]

    Epoch [84/100] Loss D: [0.64504385], Loss G: [4.5682697]


    100%|██████████| 170/170 [00:23<00:00,  7.27it/s]
      1%|          | 1/170 [00:00<00:25,  6.69it/s]

    Epoch [85/100] Loss D: [0.78593516], Loss G: [4.906456]


    100%|██████████| 170/170 [00:23<00:00,  7.32it/s]
      1%|          | 1/170 [00:00<00:25,  6.75it/s]

    Epoch [86/100] Loss D: [0.6439626], Loss G: [4.36108]


    100%|██████████| 170/170 [00:23<00:00,  7.31it/s]
      1%|          | 1/170 [00:00<00:24,  6.83it/s]

    Epoch [87/100] Loss D: [0.65820193], Loss G: [4.0977488]


      1%|          | 2/170 [00:00<00:24,  6.96it/s]100%|██████████| 170/170 [00:23<00:00,  7.30it/s]
      1%|          | 1/170 [00:00<00:24,  6.82it/s]

    Epoch [88/100] Loss D: [0.65204835], Loss G: [4.3304462]


      1%|          | 2/170 [00:00<00:24,  6.95it/s]100%|██████████| 170/170 [00:23<00:00,  7.31it/s]
      1%|          | 1/170 [00:00<00:24,  6.79it/s]

    Epoch [89/100] Loss D: [0.6524645], Loss G: [4.3844604]


    100%|██████████| 170/170 [00:23<00:00,  7.30it/s]


    Epoch [90/100] Loss D: [0.67250204], Loss G: [4.608688]


    100%|██████████| 170/170 [00:23<00:00,  7.29it/s]
      1%|          | 1/170 [00:00<00:24,  6.88it/s]

    Epoch [91/100] Loss D: [0.6917672], Loss G: [4.214923]


      1%|          | 2/170 [00:00<00:23,  7.01it/s]100%|██████████| 170/170 [00:23<00:00,  7.32it/s]
      1%|          | 1/170 [00:00<00:24,  6.99it/s]

    Epoch [92/100] Loss D: [0.63771534], Loss G: [4.2107615]


    100%|██████████| 170/170 [00:23<00:00,  7.33it/s]
      1%|          | 1/170 [00:00<00:24,  6.95it/s]

    Epoch [93/100] Loss D: [0.6473569], Loss G: [4.1955166]


    100%|██████████| 170/170 [00:23<00:00,  7.29it/s]
      1%|          | 1/170 [00:00<00:24,  6.90it/s]

    Epoch [94/100] Loss D: [0.6381893], Loss G: [4.1456714]


      1%|          | 2/170 [00:00<00:24,  7.00it/s]100%|██████████| 170/170 [00:23<00:00,  7.29it/s]
      1%|          | 1/170 [00:00<00:24,  6.92it/s]

    Epoch [95/100] Loss D: [0.6624663], Loss G: [3.7003992]


      2%|▏         | 4/170 [00:00<00:23,  7.20it/s]100%|██████████| 170/170 [00:23<00:00,  7.29it/s]
      1%|          | 1/170 [00:00<00:24,  6.94it/s]

    Epoch [96/100] Loss D: [0.66718966], Loss G: [4.3378816]


      1%|          | 2/170 [00:00<00:23,  7.04it/s]100%|██████████| 170/170 [00:23<00:00,  7.28it/s]
      1%|          | 1/170 [00:00<00:24,  6.88it/s]

    Epoch [97/100] Loss D: [0.7393694], Loss G: [4.2994103]


      1%|          | 2/170 [00:00<00:24,  6.99it/s]100%|██████████| 170/170 [00:23<00:00,  7.29it/s]
      1%|          | 1/170 [00:00<00:24,  6.90it/s]

    Epoch [98/100] Loss D: [0.6694163], Loss G: [4.1130667]


      2%|▏         | 3/170 [00:00<00:23,  7.12it/s]100%|██████████| 170/170 [00:23<00:00,  7.32it/s]
      1%|          | 1/170 [00:00<00:24,  6.88it/s]

    Epoch [99/100] Loss D: [0.68636537], Loss G: [4.5008516]


      1%|          | 2/170 [00:00<00:24,  7.00it/s]100%|██████████| 170/170 [00:23<00:00,  7.28it/s]


    Epoch [100/100] Loss D: [0.6598151], Loss G: [4.105522]


## 最后：用你补齐的代码试试卡通化的效果吧！


```python
# 为生成器加载权重
results_save_path = 'work/results'

weights_save_path = 'work/weights'
last_weights_path = os.path.join(weights_save_path, sorted(os.listdir(weights_save_path))[-1])
print('加载权重:', last_weights_path)

model_state_dict = paddle.load(last_weights_path)
generator.load_dict(model_state_dict)
generator.eval()
```

    加载权重: work/weights/epoch100.pdparams



```python
# 读取数据
img_name='data/cartoon_A2B/test/01462.png'
img_A2B = cv2.imread(img_name)
img_A = img_A2B[:, :256]                                  # 真人照
img_B = img_A2B[:, 256:]                                  # 卡通图

g_input = img_A.astype('float32') / 127.5 - 1             # 归一化
g_input = g_input[np.newaxis, ...].transpose(0, 3, 1, 2)  # NHWC -> NCHW
g_input = paddle.to_tensor(g_input)                       # numpy -> tensor

g_output = generator(g_input)
g_output = g_output.detach().numpy()                      # tensor -> numpy
g_output = g_output.transpose(0, 2, 3, 1)[0]              # NCHW -> NHWC
g_output = g_output * 127.5 + 127.5                       # 反归一化
g_output = g_output.astype(np.uint8)

img_show = np.hstack([img_A, g_output])[:,:,::-1]
plt.figure(figsize=(8, 8))
plt.imshow(img_show)
plt.show()
```


![png](output_19_0.png)

