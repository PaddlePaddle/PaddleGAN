# 老北京城影像修复

完整项目见：https://aistudio.baidu.com/aistudio/projectdetail/1796293

本项目运用[PaddleGAN](https://github.com/PaddlePaddle/PaddleGAN)实现了百年前老北京城视频的复原，其中将详细讲解如何运用视频的上色、超分辨率（提高清晰度）、插帧（提高流畅度）等AI修复技术，让那些先辈们的一举一动，一颦一簇都宛若眼前之人。

当然，如果大家觉得这个项目有趣好用的话，希望大家能够为我们[PaddleGAN](https://github.com/PaddlePaddle/PaddleGAN)的[Github主页](https://github.com/PaddlePaddle/PaddleGAN)点Star噢~

<div align='center'>
  <img src='https://ai-studio-static-online.cdn.bcebos.com/47cea097a0284dd39fc2804a53aa8ee6dad16ffe104641258046eb05af49cd64' width='1000'/>
</div>


</br>

<div align='center'>
  <img src='https://ai-studio-static-online.cdn.bcebos.com/99da82cb4c0143dfa45e40c5eb13dd4543ab55daf70f4313b81eef434d1a1ff7' width='800'/>
</div>


## 安装PaddleGAN

PaddleGAN的安装目前支持Clone GitHub和Gitee两种方式：


```python
# 安装ppgan
# 当前目录在: /home/aistudio/, 这个目录也是左边文件和文件夹所在的目录
# 克隆最新的PaddleGAN仓库到当前目录
# !git clone https://github.com/PaddlePaddle/PaddleGAN.git
# 如果从github下载慢可以从gitee clone：
!git clone https://gitee.com/paddlepaddle/PaddleGAN.git
%cd PaddleGAN/
!pip install -v -e .
```

## PaddleGAN中要使用的预测模型介绍

### 补帧模型DAIN

DAIN 模型通过探索深度的信息来显式检测遮挡。并且开发了一个深度感知的流投影层来合成中间流。在视频补帧方面有较好的效果。

```
ppgan.apps.DAINPredictor(
                        output_path='output',
                        weight_path=None,
                        time_step=None,
                        use_gpu=True,
                        remove_duplicates=False)
```

#### 参数

- `output_path (str，可选的)`: 输出的文件夹路径，默认值：`output`.
- `weight_path (None，可选的)`: 载入的权重路径，如果没有设置，则从云端下载默认的权重到本地。默认值：`None`。
- `time_step (int)`: 补帧的时间系数，如果设置为0.5，则原先为每秒30帧的视频，补帧后变为每秒60帧。
- `remove_duplicates (bool，可选的)`: 是否删除重复帧，默认值：`False`.

### 上色模型DeOldifyPredictor

DeOldify 采用自注意力机制的生成对抗网络，生成器是一个U-NET结构的网络。在图像的上色方面有着较好的效果。

```
ppgan.apps.DeOldifyPredictor(output='output', weight_path=None, render_factor=32)
```

#### 参数

- `output_path (str，可选的)`: 输出的文件夹路径，默认值：`output`.
- `weight_path (None，可选的)`: 载入的权重路径，如果没有设置，则从云端下载默认的权重到本地。默认值：`None`。
- `render_factor (int)`: 会将该参数乘以16后作为输入帧的resize的值，如果该值设置为32，
      则输入帧会resize到(32 * 16, 32 * 16)的尺寸再输入到网络中。

### 上色模型DeepRemasterPredictor

DeepRemaster 模型基于时空卷积神经网络和自注意力机制。并且能够根据输入的任意数量的参考帧对图片进行上色。

```
ppgan.apps.DeepRemasterPredictor(
                                output='output',
                                weight_path=None,
                                colorization=False,
                                reference_dir=None,
                                mindim=360):
```

#### 参数

- `output_path (str，可选的)`: 输出的文件夹路径，默认值：`output`.
- `weight_path (None，可选的)`: 载入的权重路径，如果没有设置，则从云端下载默认的权重到本地。默认值：`None`。
- `colorization (bool)`: 是否对输入视频上色，如果选项设置为 `True` ，则参考帧的文件夹路径也必须要设置。默认值：`False`。
- `reference_dir (bool)`: 参考帧的文件夹路径。默认值：`None`。
- `mindim (bool)`: 输入帧重新resize后的短边的大小。默认值：360。

### 超分辨率模型RealSRPredictor

RealSR模型通过估计各种模糊内核以及实际噪声分布，为现实世界的图像设计一种新颖的真实图片降采样框架。基于该降采样框架，可以获取与真实世界图像共享同一域的低分辨率图像。并且提出了一个旨在提高感知度的真实世界超分辨率模型。对合成噪声数据和真实世界图像进行的大量实验表明，该模型能够有效降低了噪声并提高了视觉质量。

```
ppgan.apps.RealSRPredictor(output='output', weight_path=None)
```

#### 参数

- `output_path (str，可选的)`: 输出的文件夹路径，默认值：`output`.
- `weight_path (None，可选的)`: 载入的权重路径，如果没有设置，则从云端下载默认的权重到本地。默认值：`None`。

### 超分辨率模型EDVRPredictor

EDVR模型提出了一个新颖的视频具有增强可变形卷积的还原框架：第一，为了处理大动作而设计的一个金字塔，级联和可变形（PCD）对齐模块，使用可变形卷积以从粗到精的方式在特征级别完成对齐；第二，提出时空注意力机制（TSA）融合模块，在时间和空间上都融合了注意机制，用以增强复原的功能。

```
ppgan.apps.EDVRPredictor(output='output', weight_path=None)
```

#### 参数

- `output_path (str，可选的)`: 输出的文件夹路径，默认值：`output`.
- `weight_path (None，可选的)`: 载入的权重路径，如果没有设置，则从云端下载默认的权重到本地。默认值：`None`。

## 使用PaddleGAN进行视频修复


```python
# 导入一些可视化需要的包
import cv2
import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import warnings
warnings.filterwarnings("ignore")
```


```python
# 定义一个展示视频的函数
def display(driving, fps, size=(8, 6)):
    fig = plt.figure(figsize=size)

    ims = []
    for i in range(len(driving)):
        cols = []
        cols.append(driving[i])

        im = plt.imshow(np.concatenate(cols, axis=1), animated=True)
        plt.axis('off')
        ims.append([im])

    video = animation.ArtistAnimation(fig, ims, interval=1000.0/fps, repeat_delay=1000)

    plt.close()
    return video
```


```python
# 展示一下输入的视频, 如果视频太大，时间会非常久，可以跳过这个步骤
video_path = '/home/aistudio/Peking_input360p_clip6_5s.mp4'
video_frames = imageio.mimread(video_path, memtest=False)

# 获得视频的原分辨率
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
    

HTML(display(video_frames, fps).to_html5_video())
```


```python
# 使用插帧(DAIN), 上色(DeOldify), 超分(EDVR)这三个模型对该视频进行修复
# input参数表示输入的视频路径
# output表示处理后的视频的存放文件夹
# proccess_order 表示使用的模型和顺序（目前支持）
%cd /home/aistudio/PaddleGAN/applications/
!python tools/video-enhance.py --input /home/aistudio/Peking_input360p_clip6_5s.mp4 \
                               --process_order DAIN DeOldify EDVR \
                               --output output_dir
```


```python
# 展示一下处理好的视频, 如果视频太大，时间会非常久，可以下载下来看
# 这个路径可以查看上个code cell的最后打印的output video path
output_video_path = '/home/aistudio/PaddleGAN/applications/output_dir/EDVR/Peking_input360p_clip6_5s_deoldify_out_edvr_out.mp4'

video_frames = imageio.mimread(output_video_path, memtest=False)

# 获得视频的原分辨率
cap = cv2.VideoCapture(output_video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
    

HTML(display(video_frames, fps, size=(16, 12)).to_html5_video())
```