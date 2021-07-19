
简体中文 | [English](./README.md)

# PaddleGAN

飞桨生成对抗网络开发套件--PaddleGAN，为开发者提供经典及前沿的生成对抗网络高性能实现，并支撑开发者快速构建、训练及部署生成对抗网络，以供学术、娱乐及产业应用。

GAN--生成对抗网络，被“卷积网络之父”**Yann LeCun（杨立昆）**誉为**「过去十年计算机科学领域最有趣的想法之一」**，是近年来火遍全网，AI研究者最为关注的深度学习技术方向之一。

<div align='center'>
  <img src='./docs/imgs/ppgan.jpg'>
</div>

[![License](https://img.shields.io/badge/license-Apache%202-red.svg)](LICENSE)![python version](https://img.shields.io/badge/python-3.6+-orange.svg)



## 近期活动更新🔥🔥🔥

- 2020.12.10

  《大谷 Spitzer 手把手教你修复百年前老北京影像》b站直播中奖用户名单请点击[PaddleGAN直播中奖名单](./docs/luckydraw.md)查看~

  **想要看直播回放视频请点击链接：https://www.bilibili.com/video/BV1GZ4y1g7xc**

- 2021.4.15~4.22

  生成对抗网络七日打卡营火爆来袭，赶紧让百度资深研发带你上车GAN起来吧！

  **直播回放与课件资料：https://aistudio.baidu.com/aistudio/course/introduce/16651**

- 🔥**2021.7.9-2021.9**🔥

  **💙AI创造营：Metaverse启动机之重构现世💙**

  **PaddlePaddle × Wechaty × Mixlab 创意赛，参赛者可大开脑洞，运用PaddleGAN的花式能力，打造属于你自己的聊天机器人！**

  **奖品丰厚，等你来拿🎁🎈🎊**

  💰**一等奖 1 名**：3万元人民币 / 队

  🎮**二等奖 2 名**：PS5游戏机 1个(价值5000元) / 队

  🕶**三等奖 3 名**：VR眼镜 1个(价值3000元) / 队

  💝**最佳人气奖 1 名**：3D打印机 1个(价值2000元) / 队

  **还在等什么，快来点击报名吧：https://aistudio.baidu.com/aistudio/competition/detail/98**

  **如何用PaddleGAN在比赛中杀出重围？请见：[PaddleGAN X WeChaty Demo示例](./paddlegan-wechaty-demo/REAME.md)**

  **更多详情，请查看[比赛讲解直播回放](https://www.bilibili.com/video/BV18y4y1T7Ek)💞**

## 文档教程

### 安装

- 环境依赖：
  - PaddlePaddle >= 2.1.0
  - Python >= 3.6
  - CUDA >= 10.1
- [PaddleGAN详细安装教程](./docs/zh_CN/install.md)

### 入门教程

- [快速开始](./docs/zh_CN/get_started.md)
- [数据准备](./docs/zh_CN/data_prepare.md)
- [API接口使用文档](./docs/zh_CN/apis/apps.md)
- 配置文件说明（敬请期待）

### 产业级应用

- [智能影像修复](./docs/zh_CN/industrial_solution/video_restore_cn.md)

## 模型库

* 图像翻译
  * 风格迁移：[Pixel2Pixel](./docs/zh_CN/tutorials/pix2pix_cyclegan.md)
  * 风格迁移：[CycleGAN](./docs/zh_CN/tutorials/pix2pix_cyclegan.md)
  * 图像艺术风格转换：[LapStyle](./docs/zh_CN/tutorials/lap_style.md)
  * 人脸换妆：[PSGAN](./docs/zh_CN/tutorials/psgan.md)
  * 照片动漫化：[AnimeGANv2](./docs/zh_CN/tutorials/animegan.md)
  * 人像动漫化：[U-GAT-IT](./docs/zh_CN/tutorials/ugatit.md)
  * 人脸卡通化：[Photo2Cartoon](docs/zh_CN/tutorials/photo2cartoon.md)
* 动作迁移
  * 人脸表情迁移：[First Order Motion Model](./docs/zh_CN/tutorials/motion_driving.md)
  * 唇形合成：[Wav2Lip](docs/zh_CN/tutorials/wav2lip.md)
* 生成
  * [DCGAN](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/ppgan/models/dc_gan_model.py)
  * WGAN
  * 人脸生成：[StyleGAN2](./docs/zh_CN/tutorials/styleganv2.md)
  * 图像编码：[Pixel2Style2Pixel](./docs/zh_CN/tutorials/pixel2style2pixel.md)
* 分辨率提升
  * 单张图片超分：[Single Image Super Resolution(SISR)](./docs/zh_CN/tutorials/single_image_super_resolution.md)
  * 视频超分：[Single Image Super Resolution(SISR)](./docs/zh_CN/tutorials/single_image_super_resolution.md)
* 语义分割
  * 人脸解析：[FaceParsing](./docs/zh_CN/tutorials/face_parse.md)


## 复合应用

* [智能影像修复](./docs/zh_CN/industrial_solution/video_restore_cn.md)

## 在线教程

您可以通过[人工智能学习与实训社区AI Studio](https://aistudio.baidu.com/aistudio/index) 的示例工程在线体验PaddleGAN的部分能力:

|在线教程      |    链接   |
|--------------|-----------|
|表情动作迁移-一键实现多人版「蚂蚁呀嘿」 | [点击体验](https://aistudio.baidu.com/aistudio/projectdetail/1603391) |
|老北京视频修复|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/1161285)|
|表情动作迁移-当苏大强唱起unravel |[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/1048840)|


## 效果展示


### 风格迁移

<div align='center'>
  <img src='https://user-images.githubusercontent.com/48054808/119464966-d5c1c000-bd75-11eb-9696-9bb75357229f.gif'width='700' height='200'/>
</div>


### 老视频修复

<div align='center'>
  <img src='https://user-images.githubusercontent.com/48054808/119469496-fc81f580-bd79-11eb-865a-5e38482b1ae8.gif' width='700'/> 
</div>



### 动作迁移

<div align='center'>
  <img src='https://user-images.githubusercontent.com/48054808/119469551-0a377b00-bd7a-11eb-9117-e4871c8fb9c0.gif' width='700'/>
</div>


### 超分辨率

<div align='center'>
  <img src='https://user-images.githubusercontent.com/48054808/119469753-3e12a080-bd7a-11eb-9cde-4fa01b3201ab.png'width='700' height='250'/>
</div>



### 妆容迁移

<div align='center'>
  <img src='https://user-images.githubusercontent.com/48054808/119469834-4ff44380-bd7a-11eb-93b6-05b705dcfbf2.png'width='700' height='250'/>
</div>



### 人脸动漫化

<div align='center'>
  <img src='https://user-images.githubusercontent.com/48054808/119469952-6bf7e500-bd7a-11eb-89ad-9a78b10bd4ab.png'width='700' height='250'/>
</div>



### 写实人像卡通化

<div align='center'>
  <img src='https://user-images.githubusercontent.com/48054808/119470028-7f0ab500-bd7a-11eb-88e9-78a6b9e2e319.png'width='700' height='250'/>
</div>



### 照片动漫化

<div align='center'>
  <img src='https://user-images.githubusercontent.com/48054808/119470099-9184ee80-bd7a-11eb-8b12-c9400fe01266.png'width='700' height='250'/>
</div>



### 唇形同步

<div align='center'>
  <img src='https://user-images.githubusercontent.com/48054808/119470166-a6618200-bd7a-11eb-9f98-58052ce21b14.gif'width='700'>
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

SIG贡献:

- [zhen8838](https://github.com/zhen8838): 贡献AnimeGANv2.
- [Jay9z](https://github.com/Jay9z): 贡献DCGAN的示例、修改安装文档等。
- [HighCWu](https://github.com/HighCWu): 贡献c-DCGAN和WGAN，以及对`paddle.vision.datasets`数据集的支持；贡献inversion部分代码复现。
- [hao-qiang](https://github.com/hao-qiang) & [ minivision-ai ](https://github.com/minivision-ai): 贡献人像卡通化photo2cartoon项目。
- [lyl120117](https://github.com/lyl120117)：贡献去模糊MPRNet推理代码。


## 贡献代码

我们非常欢迎您可以为PaddleGAN提供任何贡献和建议。大多数贡献都需要同意参与者许可协议（CLA）。当提交拉取请求时，CLA机器人会自动检查您是否需要提供CLA。 只需要按照机器人提供的说明进行操作即可。CLA只需要同意一次，就能应用到所有的代码仓库上。关于更多的流程请参考[贡献指南](docs/zh_CN/contribute.md)。

## 许可证书

本项目的发布受[Apache 2.0 license](LICENSE)许可认证。
