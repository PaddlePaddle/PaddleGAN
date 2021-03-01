## 老视频修复

老视频往往具有帧数少，无色彩，分辨率低等特点。于是针对这些特点，我们使用补帧，上色，超分等模型对视频进行修复。

### 使用applications中的video-enhance.py工具进行快速开始视频修复
```
cd applications
python tools/video-enhance.py --input you_video_path.mp4 --process_order DAIN DeOldify EDVR --output output_dir
```
#### 参数

- `--input (str)`: 输入的视频路径。
- `--output (str)`: 输出的视频路径。
- `--process_order`: 调用的模型名字和顺序，比如输入为 `DAIN DeOldify EDVR`，则会顺序调用 `DAINPredictor` `DeOldifyPredictor` `EDVRPredictor` 。

#### 效果展示
![](../../imgs/color_sr_peking.gif)


### 快速体验
我们在ai studio制作了一个[ai studio 老北京视频修复教程](https://aistudio.baidu.com/aistudio/projectdetail/1161285)

### 注意事项

* 在使用本教程前，请确保您已经[安装完paddle和ppgan](../install.md)。

* 本教程的所有命令都基于PaddleGAN/applications主目录进行执行。

* 各个模型耗时较长，尤其使超分辨率模型，建议输入的视频分辨率低一些，时长短一些。

* 需要运行在gpu环境上

### ppgan提供的可用于视频修复的预测api简介
可以根据要修复的视频的特点，使用不同的模型与参数

### 补帧模型DAIN
DAIN 模型通过探索深度的信息来显式检测遮挡。并且开发了一个深度感知的流投影层来合成中间流。在视频补帧方面有较好的效果。
![](../../imgs/dain_network.png)

```
ppgan.apps.DAINPredictor(
                        output='output',
                        weight_path=None,
                        time_step=None,
                        use_gpu=True,
                        remove_duplicates=False)
```
#### 参数

- `output (str，可选的)`: 输出的文件夹路径，默认值：`output`.
- `weight_path (None，可选的)`: 载入的权重路径，如果没有设置，则从云端下载默认的权重到本地。默认值：`None`。
- `time_step (int)`: 补帧的时间系数，如果设置为0.5，则原先为每秒30帧的视频，补帧后变为每秒60帧。
- `remove_duplicates (bool，可选的)`: 是否删除重复帧，默认值：`False`.

### 上色模型DeOldifyPredictor
DeOldify 采用自注意力机制的生成对抗网络，生成器是一个U-NET结构的网络。在图像的上色方面有着较好的效果。
![](../../imgs/deoldify_network.png)

```
ppgan.apps.DeOldifyPredictor(output='output', weight_path=None, render_factor=32)
```
#### 参数

- `output (str，可选的)`: 输出的文件夹路径，默认值：`output`.
- `weight_path (None，可选的)`: 载入的权重路径，如果没有设置，则从云端下载默认的权重到本地。默认值：`None`。
- `artistic (bool)`: 是否使用偏"艺术性"的模型。"艺术性"的模型有可能产生一些有趣的颜色，但是毛刺比较多。
- `render_factor (int)`: 会将该参数乘以16后作为输入帧的resize的值，如果该值设置为32，
                         则输入帧会resize到(32 * 16, 32 * 16)的尺寸再输入到网络中。

### 上色模型DeepRemasterPredictor
DeepRemaster 模型基于时空卷积神经网络和自注意力机制。并且能够根据输入的任意数量的参考帧对图片进行上色。
![](../../imgs/remaster_network.png)

```
ppgan.apps.DeepRemasterPredictor(
                                output='output',
                                weight_path=None,
                                colorization=False,
                                reference_dir=None,
                                mindim=360):
```
#### 参数

- `output (str，可选的)`: 输出的文件夹路径，默认值：`output`.
- `weight_path (None，可选的)`: 载入的权重路径，如果没有设置，则从云端下载默认的权重到本地。默认值：`None`。
- `colorization (bool)`: 是否对输入视频上色，如果选项设置为 `True` ，则参考帧的文件夹路径也必须要设置。默认值：`False`。
- `reference_dir (bool)`: 参考帧的文件夹路径。默认值：`None`。
- `mindim (bool)`: 输入帧重新resize后的短边的大小。默认值：360。

### 超分辨率模型RealSRPredictor
RealSR模型通过估计各种模糊内核以及实际噪声分布，为现实世界的图像设计一种新颖的真实图片降采样框架。基于该降采样框架，可以获取与真实世界图像共享同一域的低分辨率图像。并且提出了一个旨在提高感知度的真实世界超分辨率模型。对合成噪声数据和真实世界图像进行的大量实验表明，该模型能够有效降低了噪声并提高了视觉质量。

![](../../imgs/realsr_network.png)

```
ppgan.apps.RealSRPredictor(output='output', weight_path=None)
```
#### 参数

- `output (str，可选的)`: 输出的文件夹路径，默认值：`output`.
- `weight_path (None，可选的)`: 载入的权重路径，如果没有设置，则从云端下载默认的权重到本地。默认值：`None`。
-
### 超分辨率模型EDVRPredictor
EDVR模型提出了一个新颖的视频具有增强可变形卷积的还原框架：第一，为了处理大动作而设计的一个金字塔，级联和可变形（PCD）对齐模块，使用可变形卷积以从粗到精的方式在特征级别完成对齐；第二，提出时空注意力机制（TSA）融合模块，在时间和空间上都融合了注意机制，用以增强复原的功能。

EDVR模型是一个基于连续帧的超分模型，能够有效利用帧间的信息，速度比RealSR模型快。

![](../../imgs/edvr_network.png)

```
ppgan.apps.EDVRPredictor(output='output', weight_path=None)
```
#### 参数

- `output (str，可选的)`: 输出的文件夹路径，默认值：`output`.
- `weight_path (None，可选的)`: 载入的权重路径，如果没有设置，则从云端下载默认的权重到本地。默认值：`None`。
