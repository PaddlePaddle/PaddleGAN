# 图片超分
针对图片分辨率提升，PaddleGAN提供了[RealSR](#RealSR)、[ESRGAN](#ESRGAN)、[LESRCNN](#LESRCNN)三种模型。接下来将介绍模型预测方式。

## RealSR

[完整模型教程](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/zh_CN/tutorials/single_image_super_resolution.md)

[RealSR](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/en_US/apis/apps.md#ppganappsrealsrpredictor)模型通过估计各种模糊内核以及实际噪声分布，为现实世界的图像设计一种新颖的真实图片降采样框架。基于该降采样框架，可以获取与真实世界图像共享同一域的低分辨率图像。并且提出了一个旨在提高感知度的真实世界超分辨率模型。对合成噪声数据和真实世界图像进行的大量实验表明，该模型能够有效降低了噪声并提高了视觉质量。

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
deep_remaster.run("docs/imgs/先烈.jpg")  #原图片所在路径
```
**2. 命令行预测**

```
!python applications/tools/video-enhance.py --input /home/aistudio/Peking_input360p_clip6_5s.mp4 \ #原视频路径
                               --process_order DeepRemaster \ #对原视频处理的顺序
                               --output output_dir #成品视频所在的路径
```
## ESRGAN

[完整模型教程](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/zh_CN/tutorials/single_image_super_resolution.md)

目前ESRGAN还未封装为API供开发者们使用，因此如需使用模型，可下载使用：

| 模型 | 数据集 | 下载地址 |
|---|---|---|
| esrgan_psnr_x4  | DIV2K | [esrgan_psnr_x4](https://paddlegan.bj.bcebos.com/models/esrgan_psnr_x4.pdparams)
| esrgan_x4  | DIV2K | [esrgan_x4](https://paddlegan.bj.bcebos.com/models/esrgan_x4.pdparams)

## LESRCNN

[完整模型教程](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/zh_CN/tutorials/single_image_super_resolution.md)

目前LESRCNN还未封装为API供开发者们使用，因此如需使用模型，可下载使用：

| 模型 | 数据集 | 下载地址 |
|---|---|---|
| lesrcnn_x4  | DIV2K | [lesrcnn_x4](https://paddlegan.bj.bcebos.com/models/lesrcnn_x4.pdparams)

### 在线项目体验
**1. [老北京城影像修复](https://aistudio.baidu.com/aistudio/projectdetail/1161285)**

**2. [PaddleGAN ❤️ 520特辑](https://aistudio.baidu.com/aistudio/projectdetail/1956943?channelType=0&channel=0)**
