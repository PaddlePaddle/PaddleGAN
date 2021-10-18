# 视频上色
针对视频上色，PaddleGAN提供两种上色模型：[DeOldify](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/en_US/apis/apps.md#ppganappsdeoldifypredictor)与[DeepRemaster](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/en_US/apis/apps.md#ppganappsdeepremasterpredictor)。

## DeOldifyPredictor

[DeOldify](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/en_US/apis/apps.md#ppganappsdeoldifypredictor)采用自注意力机制的生成对抗网络，生成器是一个U-NET结构的网络。在图像/视频的上色方面有着较好的效果。

<div align='center'>
  <img src='https://user-images.githubusercontent.com/48054808/117925538-fd526a80-b329-11eb-8924-8f2614fcd9e6.png'>
</div>

### 参数

- `output (str，可选的)`: 输出的文件夹路径，默认值：`output`.
- `weight_path (None，可选的)`: 载入的权重路径，如果没有设置，则从云端下载默认的权重到本地。默认值：`None`。
- `artistic (bool)`: 是否使用偏"艺术性"的模型。"艺术性"的模型有可能产生一些有趣的颜色，但是毛刺比较多。
- `render_factor (int)`: 会将该参数乘以16后作为输入帧的resize的值，如果该值设置为32，
                         则输入帧会resize到(32 * 16, 32 * 16)的尺寸再输入到网络中。


### 使用方式
**1. API预测**

```
from ppgan.apps import DeOldifyPredictor
deoldify = DeOldifyPredictor()
deoldify.run("/home/aistudio/Peking_input360p_clip6_5s.mp4") #原视频所在路径
```
*`run`接口为图片/视频通用接口，由于这里对象是视频，可以使用`run_video`的接口

**2. 命令行预测**

```
!python applications/tools/video-enhance.py --input /home/aistudio/Peking_input360p_clip6_5s.mp4 \ #原视频路径
                               --process_order DeOldify \ #对原视频处理的顺序
                               --output output_dir #成品视频所在的路径
```

## DeepRemasterPredictor

[DeepRemaster](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/en_US/apis/apps.md#ppganappsdeepremasterpredictor) 模型目前只能用于对视频上色，基于时空卷积神经网络和自注意力机制。并且能够根据输入的任意数量的参考帧对视频中的每一帧图片进行上色。
![](../../imgs/remaster_network.png)

<div align='center'>
  <img src='https://user-images.githubusercontent.com/48054808/117925558-05120f00-b32a-11eb-9727-d1c0d5814dc5.png'>
</div>

```
ppgan.apps.DeepRemasterPredictor(
                                output='output',
                                weight_path=None,
                                colorization=False,
                                reference_dir=None,
                                mindim=360):
```

### 参数

- `output (str，可选的)`: 输出的文件夹路径，默认值：`output`.
- `weight_path (None，可选的)`: 载入的权重路径，如果没有设置，则从云端下载默认的权重到本地。默认值：`None`。
- `colorization (bool)`: 是否对输入视频上色，如果选项设置为 `True` ，则参考帧的文件夹路径也必须要设置。默认值：`False`。
- `reference_dir (bool)`: 参考帧的文件夹路径。默认值：`None`。
- `mindim (bool)`: 输入帧重新resize后的短边的大小。默认值：360。

### 使用方式
**1. API预测**

```
from ppgan.apps import DeepRemasterPredictor
deep_remaster = DeepRemasterPredictor()
deep_remaster.run("docs/imgs/test_old.jpeg")  #原视频所在路径

```

**2. 命令行预测**

```
!python applications/tools/video-enhance.py --input /home/aistudio/Peking_input360p_clip6_5s.mp4 \ #原视频路径
                               --process_order DeepRemaster \ #对原视频处理的顺序
                               --output output_dir #成品视频所在的路径
```

### 在线项目体验
**1. [老北京城影像修复](https://aistudio.baidu.com/aistudio/projectdetail/1161285)**

**2. [PaddleGAN ❤️ 520特辑](https://aistudio.baidu.com/aistudio/projectdetail/1956943?channelType=0&channel=0)**
