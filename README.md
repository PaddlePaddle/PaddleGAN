<div align='center'>
  <img src='./docs/imgs/ppgan.jpg'>
</div>

简体中文 | [English](./README_en.md)

# PaddleGAN

PaddleGAN 是一个基于飞桨的生成对抗网络开发工具包.

### 图片变换
<div align='center'>
  <img src='./docs/imgs/A2B.png'>
</div>
<div align='center'>
  <img src='./docs/imgs/B2A.png'>
</div>

### 妆容迁移
<div align='center'>
  <img src='./docs/imgs/makeup_shifter.png'>
</div>

### 老视频修复
<div align='center'>
  <img src='./docs/imgs/color_sr_peking.gif'>
</div>

### 超分辨率
<div align='center'>
  <img src='./docs/imgs/sr_demo.png'>
</div>

### 动作驱动
<div align='center'>
  <img src='./docs/imgs/first_order.gif'>
</div>

特性:

- 高度的灵活性:

  模块化设计，解耦各个网络组件，开发者轻松搭建、试用各种检测模型及优化策略，快速得到高性能、定制化的算法。

- 丰富的应用:

  PaddleGAN 提供了非常多的应用，比如说图像生成，图像修复，图像上色，视频补帧，人脸妆容迁移等.

## 安装

请参考[安装文档](./docs/zh_CN/install.md)来进行PaddlePaddle和ppgan的安装

## 快速开始

通过ppgan.app接口使用预训练模型:

 ```python
 from ppgan.apps import RealSRPredictor
 sr = RealSRPredictor()
 sr.run("docs/imgs/monarch.png")
 ```

更多训练、评估教程参考:

- [数据准备](./docs/zh_CN/data_prepare.md)
- [训练/评估/推理教程](./docs/zh_CN/get_started.md)

## 模型教程

* [Pixel2Pixel](./docs/zh_CN/tutorials/pix2pix_cyclegan.md)
* [CycleGAN](./docs/zh_CN/tutorials/pix2pix_cyclegan.md)
* [PSGAN](./docs/zh_CN/tutorials/psgan.md)
* [First Order Motion Model](./docs/zh_CN/tutorials/motion_driving.md)
* [视频修复](./docs/zh_CN/tutorials/video_restore.md)

## 在线体验

通过[AI Studio实训平台](https://aistudio.baidu.com/aistudio/index)在线体验:

|在线教程      |    链接   |
|--------------|-----------|
|老北京视频修复|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/1161285)|
|表情动作迁移-当苏大强唱起unravel |[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/1048840)|


## 版本更新

- v0.1.0 (2020.11.02)
  - 初版发布，支持Pixel2Pixel、CycleGAN、PSGAN模型，支持视频插针、超分、老照片/视频上色、视频动作生成等应用。
  - 模块化设计，接口简单易用。


## 贡献代码

我们非常欢迎您可以为PaddleGAN提供任何贡献和建议。大多数贡献都需要同意参与者许可协议（CLA）。当提交拉取请求时，CLA机器人会自动检查您是否需要提供CLA。 只需要按照机器人提供的说明进行操作即可。CLA只需要同意一次，就能应用到所有的代码仓库上。关于更多的流程请参考[贡献指南](docs/zh_CN/contribute.md)。

## 许可证书
本项目的发布受[Apache 2.0 license](LICENSE)许可认证。
