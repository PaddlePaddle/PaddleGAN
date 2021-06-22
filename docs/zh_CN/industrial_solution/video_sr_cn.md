# 视频分辨率提升

针对视频超分，PaddleGAN提供了两种模型，[RealSR](#RealSR)、[EDVR](#EDVR)。

## RealSR

[完整模型教程](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/zh_CN/tutorials/single_image_super_resolution.md)

[RealSR](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/zh_CN/apis/apps.md#ppganappsrealsrpredictor)模型通过估计各种模糊内核以及实际噪声分布，为现实世界的图像设计一种新颖的真实图片降采样框架。基于该降采样框架，可以获取与真实世界图像共享同一域的低分辨率图像。并且提出了一个旨在提高感知度的真实世界超分辨率模型。对合成噪声数据和真实世界图像进行的大量实验表明，该模型能够有效降低了噪声并提高了视觉质量。

<div align='center'>
  <img src='https://user-images.githubusercontent.com/48054808/117925551-02afb500-b32a-11eb-9a11-14e484daa953.png'>
</div>

```
ppgan.apps.RealSRPredictor(output='output', weight_path=None)
```
### 参数

- `output (str，可选的)`: 输出的文件夹路径，默认值：`output`.
- `weight_path (None，可选的)`: 载入的权重路径，如果没有设置，则从云端下载默认的权重到本地。默认值：`None`。


### 使用方式
**1. API预测**

```
from ppgan.apps import DeepRemasterPredictor
deep_remaster = DeepRemasterPredictor()
deep_remaster.run("/home/aistudio/Peking_input360p_clip6_5s.mp4")  #原视频所在路径

```

**2. 命令行预测**

```
!python applications/tools/video-enhance.py --input /home/aistudio/Peking_input360p_clip6_5s.mp4 \ #原视频路径
                               --process_order DeepRemaster \ #对原视频处理的顺序
                               --output output_dir #成品视频所在的路径
```


## EDVR

[完整模型教程](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/zh_CN/tutorials/video_super_resolution.md)

[EDVR](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/zh_CN/apis/apps.md#ppganappsedvrpredictor)模型提出了一个新颖的视频具有增强可变形卷积的还原框架：第一，为了处理大动作而设计的一个金字塔，级联和可变形（PCD）对齐模块，使用可变形卷积以从粗到精的方式在特征级别完成对齐；第二，提出时空注意力机制（TSA）融合模块，在时间和空间上都融合了注意机制，用以增强复原的功能。

[EDVR](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/zh_CN/apis/apps.md#ppganappsedvrpredictor)模型是一个基于连续帧的超分模型，能够有效利用帧间的信息，速度比[RealSR](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/zh_CN/apis/apps.md#ppganappsrealsrpredictor)模型快。

<div align='center'>
  <img src='https://user-images.githubusercontent.com/48054808/117925546-004d5b00-b32a-11eb-9af9-3b19d666de01.png'>
</div>

```
ppgan.apps.EDVRPredictor(output='output', weight_path=None)
```

### 参数

- `output (str，可选的)`: 输出的文件夹路径，默认值：`output`.
- `weight_path (None，可选的)`: 载入的权重路径，如果没有设置，则从云端下载默认的权重到本地。默认值：`None`。


### 使用方式
**1. API预测** 

目前API预测方式只支持在静态图下运行，需加上启动静态图命令，后续会支持动态图，敬请期待~

```
paddle.enable_static()

from ppgan.apps import EDVRPredictor
sr = EDVRPredictor()
# 测试一个视频文件
sr.run("/home/aistudio/Peking_input360p_clip6_5s.mp4")  #原视频所在路径

paddle.disable_static()

```

**2. 命令行预测**

```
!python applications/tools/video-enhance.py --input /home/aistudio/Peking_input360p_clip6_5s.mp4 \ #原视频路径
                               --process_order EDVR \ #对原视频处理的顺序，此处注意“EDVR”四个字母都需大写
                               --output output_dir #成品视频所在的路径
```

### 在线项目体验
**1. [老北京城影像修复](https://aistudio.baidu.com/aistudio/projectdetail/1161285)**

**2. [PaddleGAN ❤️ 520特辑](https://aistudio.baidu.com/aistudio/projectdetail/1956943?channelType=0&channel=0)**
