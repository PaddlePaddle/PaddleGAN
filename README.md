
简体中文 | [English](./README_en.md)

# PaddleGAN

飞桨生成对抗网络开发套件--PaddleGAN，为开发者提供经典及前沿的生成对抗网络高性能实现，并支撑开发者快速构建、训练及部署生成对抗网络，以供学术、娱乐及产业应用。

GAN--生成对抗网络，被“卷积网络之父”**Yann LeCun（杨立昆）**誉为**「过去十年计算机科学领域最有趣的想法之一」**，是近年来火遍全网，AI研究者最为关注的深度学习技术方向之一。

<div align='center'>
  <img src='./docs/imgs/ppgan.jpg'>
</div>

[![License](https://img.shields.io/badge/license-Apache%202-red.svg)](LICENSE)![python version](https://img.shields.io/badge/python-3.6+-orange.svg)

## 快速开始

* 请确保您按照[安装文档](./docs/zh_CN/install.md)的说明正确安装了PaddlePaddle和PaddleGAN

* 通过ppgan.apps接口使用预训练模型:

 ```python
 from ppgan.apps import RealSRPredictor
 sr = RealSRPredictor()
 sr.run("docs/imgs/monarch.png")
 ```
* 更多预训练模型的使用请参考[ppgan.apps apis](./docs/zh_CN/apis/apps.md)
* 更多训练、评估教程:
  * [数据准备](./docs/zh_CN/data_prepare.md)
  * [训练/评估/推理教程](./docs/zh_CN/get_started.md)

## 经典模型实现

* [Pixel2Pixel](./docs/zh_CN/tutorials/pix2pix_cyclegan.md)
* [CycleGAN](./docs/zh_CN/tutorials/pix2pix_cyclegan.md)
* [PSGAN](./docs/zh_CN/tutorials/psgan.md)
* [First Order Motion Model](./docs/zh_CN/tutorials/motion_driving.md)
* [FaceParsing](./docs/zh_CN/tutorials/face_parse.md)

## 复合应用

* [视频修复](./docs/zh_CN/tutorials/video_restore.md)

## 在线教程

您可以通过[人工智能学习与实训社区AI Studio](https://aistudio.baidu.com/aistudio/index) 的示例工程在线体验PaddleGAN的部分能力:

|在线教程      |    链接   |
|--------------|-----------|
|老北京视频修复|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/1161285)|
|表情动作迁移-当苏大强唱起unravel |[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/1048840)|

## 效果展示

### 图片变换

<div align='center'>
  <img src='./docs/imgs/horse2zebra.gif'width='700' height='200'/>
</div>

### 老视频修复

<div align='center'>
  <img src='./docs/imgs/color_sr_peking.gif' width='700'/>
</div>


### 动作迁移

<div align='center'>
  <img src='./docs/imgs/first_order.gif' width='700'/>
</div>

### 超分辨率

<div align='center'>
  <img src='./docs/imgs/sr_demo.png'width='700' height='250'/>
</div>


### 妆容迁移

<div align='center'>
  <img src='./docs/imgs/makeup_shifter.png'width='700' height='250'/>
</div>


## 版本更新

- v0.1.0 (2020.11.02)
  - 初版发布，支持Pixel2Pixel、CycleGAN、PSGAN模型，支持视频插针、超分、老照片/视频上色、视频动作生成等应用。
  - 模块化设计，接口简单易用。

## 欢迎加入PaddleGAN技术交流群

扫描二维码加入PaddleGAN QQ群[群号：1058398620]，获得更高效的问题答疑，与各行业开发者交流讨论，我们期待您的加入！

<div align='center'>
  <img src='./docs/imgs/qq.png'width='250' height='300'/>
</div>
### PaddleGAN 特别兴趣小组（Special Interest Group）

最早于1961年被[ACM（Association for Computing Machinery)](https://en.wikipedia.org/wiki/Association_for_Computing_Machinery)首次提出并使用，国际顶尖开源组织包括[Kubernates](https://kubernetes.io/)都采用SIGs的形式，使拥有同样特定兴趣的成员可以共同分享、学习知识并进行项目开发。这些成员不需要在同一国家/地区、同一个组织，只要大家志同道合，都可以奔着相同的目标一同学习、工作、玩耍~

PaddleGAN SIG就是这样一个汇集对GAN感兴趣小伙伴们的开发者组织，在这里，有百度飞桨的一线开发人员、有来自世界500强的资深工程师、有国内外顶尖高校的学生。

我们正在持续招募有兴趣、有能力的开发者加入我们一起共同建设本项目，并一起探索更多有用、有趣的应用。欢迎大家在加入群后联系我们讨论加入SIG并参与共建事宜。

## 贡献代码

我们非常欢迎您可以为PaddleGAN提供任何贡献和建议。大多数贡献都需要同意参与者许可协议（CLA）。当提交拉取请求时，CLA机器人会自动检查您是否需要提供CLA。 只需要按照机器人提供的说明进行操作即可。CLA只需要同意一次，就能应用到所有的代码仓库上。关于更多的流程请参考[贡献指南](docs/zh_CN/contribute.md)。

## 许可证书

本项目的发布受[Apache 2.0 license](LICENSE)许可认证。
