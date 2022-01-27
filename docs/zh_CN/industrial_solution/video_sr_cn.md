# 视频分辨率提升

针对视频超分，PaddleGAN提供了七种模型，[RealSR](#RealSR)、[PPMSVSR](#PPMSVSR)、[PPMSVSRLarge](#PPMSVSRLarge)、[EDVR](#EDVR)、[BasicVSR](#BasicVSR)、[IconVSR](#IconVSR)、[BasiVSRPlusPlus](#BasiVSRPlusPlus)。

## RealSR

[完整模型教程](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/zh_CN/tutorials/single_image_super_resolution.md)

[RealSR](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/en_US/apis/apps.md#ppganappsrealsrpredictor)是图像超分模式，其通过估计各种模糊内核以及实际噪声分布，为现实世界的图像设计一种新颖的真实图片降采样框架。基于该降采样框架，可以获取与真实世界图像共享同一域的低分辨率图像。并且提出了一个旨在提高感知度的真实世界超分辨率模型。对合成噪声数据和真实世界图像进行的大量实验表明，该模型能够有效降低了噪声并提高了视觉质量。

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


## PPMSVSR

[完整模型教程](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/zh_CN/tutorials/video_super_resolution.md)

[PPMSVSR](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/en_US/apis/apps.md#ppganappsppmsvsrpredictor)为PaddleGAN自研的轻量视频超分模型，是一种多阶段视频超分深度架构，具有局部融合模块、辅助损失和细化对齐模块，以逐步细化增强结果。具体来说，在第一阶段设计了局部融合模块，在特征传播之前进行局部特征融合, 以加强特征传播中跨帧特征的融合。在第二阶段中引入了一个辅助损失，使传播模块获得的特征保留了更多与HR空间相关的信息。在第三阶段中引入了一个细化的对齐模块，以充分利用前一阶段传播模块的特征信息。大量实验证实，PP-MSVSR在Vid4数据集性能优异，仅使用 1.45M 参数PSNR指标即可达到28.13dB。

[PPMSVSR](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/en_US/apis/apps.md#ppganappsppmsvsrpredictor)模型是一个轻量视频超分模型，在当前轻量视频超分模型（模型参数量小于6M）中，PPMSVSR以最小的参数量在4个常用视频超分测试数据集Vimeo90K、Vid4、UDM10和REDS4上达到最优超分效果。

<div align='center'>
  <img src='https://user-images.githubusercontent.com/79366697/145384020-a98c74df-a3b4-4477-a071-23605739ce80.png'>
</div>

```
ppgan.apps.PPMSVSRPredictor(output='output', weight_path=None, num_frames=10)
```

### 参数

- `output (str，可选的)`: 输出的文件夹路径，默认值：`output`.
- `weight_path (None，可选的)`: 载入的权重路径，如果没有设置，则从云端下载默认的权重到本地。默认值：`None`。
- `num_frames (int，可选的)`: 模型输入帧数，默认值：`10`。模型输入帧数设置的越大，模型超分效果越好.

### 使用方式
**1. API预测**

```
from ppgan.apps import PPMSVSRPredictor
sr = PPMSVSRPredictor()
# 测试一个视频文件
sr.run("/home/aistudio/Peking_input360p_clip6_5s.mp4")  #原视频所在路径

```

**2. 命令行预测**

```
!python applications/tools/video-enhance.py --input /home/aistudio/Peking_input360p_clip6_5s.mp4 \ #原视频路径
                               --process_order PPMSVSR \ #对原视频处理的顺序，此处注意“EDVR”四个字母都需大写
                               --output output_dir #成品视频所在的路径
```

## PPMSVSRLarge

[完整模型教程](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/zh_CN/tutorials/video_super_resolution.md)

[PPMSVSRLarge](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/en_US/apis/apps.md#ppganappsppmsvsrlargepredictor)为PaddleGAN自研的高精度超分模型，是一种多阶段视频超分深度架构，具有局部融合模块、辅助损失和细化对齐模块，以逐步细化增强结果。具体来说，在第一阶段设计了局部融合模块，在特征传播之前进行局部特征融合, 以加强特征传播中跨帧特征的融合。在第二阶段中引入了一个辅助损失，使传播模块获得的特征保留了更多与HR空间相关的信息。在第三阶段中引入了一个细化的对齐模块，以充分利用前一阶段传播模块的特征信息。

[PPMSVSRLarge](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/en_US/apis/apps.md#ppganappsppmsvsrlargepredictor)模型是为满足精度提升，对PPMSVSR通过增加基础快数量而构造的一个大模型。PPMSVSRLarge与当前精度最高的BasicVSR++模型相比，以相似的参数量达到了更高的精度。

```
ppgan.apps.PPMSVSRLargePredictor(output='output', weight_path=None, num_frames=10)
```

### 参数

- `output (str，可选的)`: 输出的文件夹路径，默认值：`output`.
- `weight_path (None，可选的)`: 载入的权重路径，如果没有设置，则从云端下载默认的权重到本地。默认值：`None`。
- `num_frames (int，可选的)`: 模型输入帧数，默认值：`10`。模型输入帧数设置的越大，模型超分效果越好.

### 使用方式
**1. API预测**

```
from ppgan.apps import PPMSVSRLargePredictor
sr = PPMSVSRLargePredictor()
# 测试一个视频文件
sr.run("/home/aistudio/Peking_input360p_clip6_5s.mp4")  #原视频所在路径

```

**2. 命令行预测**

```
!python applications/tools/video-enhance.py --input /home/aistudio/Peking_input360p_clip6_5s.mp4 \ #原视频路径
                               --process_order PPMSVSRLarge \ #对原视频处理的顺序，此处注意“EDVR”四个字母都需大写
                               --output output_dir #成品视频所在的路径
```



## EDVR

[完整模型教程](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/zh_CN/tutorials/video_super_resolution.md)

[EDVR](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/en_US/apis/apps.md#ppganappsedvrpredictor)模型提出了一个新颖的视频具有增强可变形卷积的还原框架：第一，为了处理大动作而设计的一个金字塔，级联和可变形（PCD）对齐模块，使用可变形卷积以从粗到精的方式在特征级别完成对齐；第二，提出时空注意力机制（TSA）融合模块，在时间和空间上都融合了注意机制，用以增强复原的功能。

[EDVR](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/en_US/apis/apps.md#ppganappsedvrpredictor)模型是一个基于连续帧的超分模型，能够有效利用帧间的信息，速度比[RealSR](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/en_US/apis/apps.md#ppganappsrealsrpredictor)模型快。

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

```
from ppgan.apps import EDVRPredictor
sr = EDVRPredictor()
# 测试一个视频文件
sr.run("/home/aistudio/Peking_input360p_clip6_5s.mp4")  #原视频所在路径

```

**2. 命令行预测**

```
!python applications/tools/video-enhance.py --input /home/aistudio/Peking_input360p_clip6_5s.mp4 \ #原视频路径
                               --process_order EDVR \ #对原视频处理的顺序，此处注意“EDVR”四个字母都需大写
                               --output output_dir #成品视频所在的路径
```

## BasicVSR

[完整模型教程](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/zh_CN/tutorials/video_super_resolution.md)

[BasicVSR](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/en_US/apis/apps.md#ppganappsbasicvsrpredictor)在VSR的指导下重新考虑了四个基本模块（即传播、对齐、聚合和上采样）的一些最重要的组件。 通过添加一些小设计，重用一些现有组件，得到了简洁的 BasicVSR。与许多最先进的算法相比，BasicVSR在速度和恢复质量方面实现了有吸引力的改进。

```
ppgan.apps.BasicVSRPredictor(output='output', weight_path=None, num_frames=10)
```

### 参数

- `output (str，可选的)`: 输出的文件夹路径，默认值：`output`.
- `weight_path (None，可选的)`: 载入的权重路径，如果没有设置，则从云端下载默认的权重到本地。默认值：`None`。
- `num_frames (int，可选的)`: 模型输入帧数，默认值：`10`。模型输入帧数设置的越大，模型超分效果越好.

### 使用方式
**1. API预测**

```
from ppgan.apps import BasicVSRPredictor
sr = BasicVSRPredictor()
# 测试一个视频文件
sr.run("/home/aistudio/Peking_input360p_clip6_5s.mp4")  #原视频所在路径

```

**2. 命令行预测**

```
!python applications/tools/video-enhance.py --input /home/aistudio/Peking_input360p_clip6_5s.mp4 \ #原视频路径
                               --process_order BasicVSR \ #对原视频处理的顺序，此处注意“EDVR”四个字母都需大写
                               --output output_dir #成品视频所在的路径
```

## IconVSR

[完整模型教程](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/zh_CN/tutorials/video_super_resolution.md)

[IconVSR](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/en_US/apis/apps.md#ppganappsiconvsrpredictor)是由BasicVSR扩展而来，其是在BasicVSR基础之上，通过添加信息重新填充机制和耦合传播方案以促进信息聚合。与BasicVSR相比，IconVSR提升了一点精度。

```
ppgan.apps.IconVSRPredictor(output='output', weight_path=None, num_frames=10)
```

### 参数

- `output (str，可选的)`: 输出的文件夹路径，默认值：`output`.
- `weight_path (None，可选的)`: 载入的权重路径，如果没有设置，则从云端下载默认的权重到本地。默认值：`None`。
- `num_frames (int，可选的)`: 模型输入帧数，默认值：`10`。模型输入帧数设置的越大，模型超分效果越好.

### 使用方式
**1. API预测**

```
from ppgan.apps import IconVSRPredictor
sr = IconVSRPredictor()
# 测试一个视频文件
sr.run("/home/aistudio/Peking_input360p_clip6_5s.mp4")  #原视频所在路径

```

**2. 命令行预测**

```
!python applications/tools/video-enhance.py --input /home/aistudio/Peking_input360p_clip6_5s.mp4 \ #原视频路径
                               --process_order IconVSR \ #对原视频处理的顺序，此处注意“EDVR”四个字母都需大写
                               --output output_dir #成品视频所在的路径
```

## BasiVSRPlusPlus

[完整模型教程](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/zh_CN/tutorials/video_super_resolution.md)

[BasiVSRPlusPlus](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/en_US/apis/apps.md#ppganappsbasicvsrpluspluspredictor)通过提出二阶网格传播和导流可变形对齐来重新设计BasicVSR。通过增强传播和对齐来增强循环框架，BasicVSR++可以更有效地利用未对齐视频帧的时空信息。 在类似的计算约束下，新组件可提高性能。特别是，BasicVSR++ 以相似的参数数量在 PSNR 方面比 BasicVSR 高0.82dB。BasicVSR++ 在NTIRE2021的视频超分辨率和压缩视频增强挑战赛中获得三名冠军和一名亚军。

<div align='center'>
  <img src='https://user-images.githubusercontent.com/79366697/145386802-5533f3df-a52b-4917-aa72-20e91833f53c.jpg'>
</div>

```
ppgan.apps.BasiVSRPlusPlusPredictor(output='output', weight_path=None, num_frames=10)
```

### 参数

- `output (str，可选的)`: 输出的文件夹路径，默认值：`output`.
- `weight_path (None，可选的)`: 载入的权重路径，如果没有设置，则从云端下载默认的权重到本地。默认值：`None`。
- `num_frames (int，可选的)`: 模型输入帧数，默认值：`10`。模型输入帧数设置的越大，模型超分效果越好.

### 使用方式
**1. API预测**

```
from ppgan.apps import BasiVSRPlusPlusPredictor
sr = BasiVSRPlusPlusPredictor()
# 测试一个视频文件
sr.run("/home/aistudio/Peking_input360p_clip6_5s.mp4")  #原视频所在路径

```

**2. 命令行预测**

```
!python applications/tools/video-enhance.py --input /home/aistudio/Peking_input360p_clip6_5s.mp4 \ #原视频路径
                               --process_order BasiVSRPlusPlus \ #对原视频处理的顺序，此处注意“EDVR”四个字母都需大写
                               --output output_dir #成品视频所在的路径
```


### 在线项目体验
**1. [PaddleGAN SOTA算法：视频超分模型PP-MSVSR详解及应用](https://aistudio.baidu.com/aistudio/projectdetail/3205183)**

**2. [老北京城影像修复](https://aistudio.baidu.com/aistudio/projectdetail/1161285)**

**3. [PaddleGAN ❤️ 520特辑](https://aistudio.baidu.com/aistudio/projectdetail/1956943?channelType=0&channel=0)**
