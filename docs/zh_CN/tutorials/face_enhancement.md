# 人脸增强

## 1. 人脸增强简介

从严重退化的人脸图像中恢复出人脸是一个非常具有挑战性的问题。由于问题的严重性和复杂的未知退化，直接训练深度神经网络通常无法得到可接受的结果。现有的基于生成对抗网络 (GAN) 的方法可以产生更好的结果，但往往会产生过度平滑的恢复。这里我们提供[GPEN](https://arxiv.org/abs/2105.06070)模型来进行人脸增强。GPEN模型首先学习用于生成高质量人脸图像的GAN并将其嵌入到U形DNN作为先验解码器，然后使用一组合成的低质量人脸图像对GAN先验嵌入DNN进行微调。 GAN 模块的设计是为了确保输入到 GAN 的隐码和噪声可以分别从 DNN 的深层和浅层特征中生成，控制重建图像的全局人脸结构、局部人脸细节和背景。所提出的 GAN 先验嵌入网络 (GPEN) 易于实现，并且可以生成视觉上逼真的结果。实验表明，GPEN 在数量和质量上都比最先进的 BFR 方法取得了显着优越的结果，特别是对于野外严重退化的人脸图像的恢复。

## 使用方法

### 人脸增强

用户使用如下代码进行人脸增强，选择本地图像作为输入：

```python
import paddle
from ppgan.faceutils.face_enhancement import FaceEnhancement

faceenhancer = FaceEnhancement()
img = faceenhancer.enhance_from_image(img)
```

注意：请将图片转为float类型输入，目前不支持int8类型

### 训练(TODO)

未来还将添加训练脚本方便用户训练出更多类型的 GPEN 人脸增强。

## 人脸增强结果展示

![1](https://user-images.githubusercontent.com/79366697/146891109-d204497f-7e71-4899-bc65-e1b101ce6293.jpg)

## 参考文献

```
@inproceedings{inproceedings,
author = {Yang, Tao and Ren, Peiran and Xie, Xuansong and Zhang, Lei},
year = {2021},
month = {06},
pages = {672-681},
title = {GAN Prior Embedded Network for Blind Face Restoration in the Wild},
doi = {10.1109/CVPR46437.2021.00073}
}

```
