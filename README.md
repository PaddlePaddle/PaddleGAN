
English | [简体中文](./README_cn.md)

# PaddleGAN

PaddleGAN provides developers with high-performance implementation of classic and SOTA Generative Adversarial Networks, and supports developers to quickly build, train and deploy GANs for academic, entertainment and industrial usage.

GAN-Generative Adversarial Network, was praised by "the Father of Convolutional Networks"  **Yann LeCun (Yang Likun)**  as **[One of the most interesting ideas in the field of computer science in the past decade]**. It's the one research area in deep learning that AI researchers are most concerned about.

<div align='center'>
  <img src='./docs/imgs/ppgan.jpg'>
</div>

[![License](https://img.shields.io/badge/license-Apache%202-red.svg)](LICENSE)![python version](https://img.shields.io/badge/python-3.6+-orange.svg)


## Recent Contributors
[![](https://sourcerer.io/fame/LaraStuStu/paddlepaddle/paddlegan/images/0)](https://sourcerer.io/fame/LaraStuStu/paddlepaddle/paddlegan/links/0)[![](https://sourcerer.io/fame/LaraStuStu/paddlepaddle/paddlegan/images/1)](https://sourcerer.io/fame/LaraStuStu/paddlepaddle/paddlegan/links/1)[![](https://sourcerer.io/fame/LaraStuStu/paddlepaddle/paddlegan/images/2)](https://sourcerer.io/fame/LaraStuStu/paddlepaddle/paddlegan/links/2)[![](https://sourcerer.io/fame/LaraStuStu/paddlepaddle/paddlegan/images/3)](https://sourcerer.io/fame/LaraStuStu/paddlepaddle/paddlegan/links/3)[![](https://sourcerer.io/fame/LaraStuStu/paddlepaddle/paddlegan/images/4)](https://sourcerer.io/fame/LaraStuStu/paddlepaddle/paddlegan/links/4)[![](https://sourcerer.io/fame/LaraStuStu/paddlepaddle/paddlegan/images/5)](https://sourcerer.io/fame/LaraStuStu/paddlepaddle/paddlegan/links/5)[![](https://sourcerer.io/fame/LaraStuStu/paddlepaddle/paddlegan/images/6)](https://sourcerer.io/fame/LaraStuStu/paddlepaddle/paddlegan/links/6)[![](https://sourcerer.io/fame/LaraStuStu/paddlepaddle/paddlegan/images/7)](https://sourcerer.io/fame/LaraStuStu/paddlepaddle/paddlegan/links/7)

## Quick Start

* Please refer to the [installation document](./docs/en_US/install.md) to make sure you have installed PaddlePaddle and PaddleGAN correctly.

* Get started through ppgan.app interface:

   ```python
   from ppgan.apps import RealSRPredictor
   sr = RealSRPredictor()
   sr.run("docs/imgs/monarch.png")
   ```
* More applications, please refer to [ppgan.apps apis](./docs/en_US/apis/apps.md)
* More tutorials:
  - [Data preparation](./docs/en_US/data_prepare.md)
  - [Training/Evaluating/Testing basic usage](./docs/zh_CN/get_started.md)

## Model Tutorial

* [Pixel2Pixel](./docs/en_US/tutorials/pix2pix_cyclegan.md)
* [CycleGAN](./docs/en_US/tutorials/pix2pix_cyclegan.md)
* [LapStyle](./docs/en_US/tutorials/lap_style.md)
* [PSGAN](./docs/en_US/tutorials/psgan.md)
* [First Order Motion Model](./docs/en_US/tutorials/motion_driving.md)
* [FaceParsing](./docs/en_US/tutorials/face_parse.md)
* [AnimeGANv2](./docs/en_US/tutorials/animegan.md)
* [U-GAT-IT](./docs/en_US/tutorials/ugatit.md)
* [Photo2Cartoon](./docs/en_US/tutorials/photo2cartoon.md)
* [Wav2Lip](./docs/en_US/tutorials/wav2lip.md)
* [Single Image Super Resolution(SISR)](./docs/en_US/tutorials/single_image_super_resolution.md)
* [Video Super Resolution(VSR)](./docs/en_US/tutorials/video_super_resolution.md)
* [StyleGAN2](./docs/en_US/tutorials/styleganv2.md)
* [Pixel2Style2Pixel](./docs/en_US/tutorials/pixel2style2pixel.md)


## Composite Application

* [Video restore](./docs/en_US/tutorials/video_restore.md)

## Online Tutorial

You can run those projects in the [AI Studio](https://aistudio.baidu.com/aistudio/projectoverview/public/1?kw=paddlegan) to learn how to use the models above:

|Online Tutorial      |    link  |
|--------------|-----------|
|Motion Driving-multi-personal "Mai-ha-hi" | [Click and Try](https://aistudio.baidu.com/aistudio/projectdetail/1603391) |
|Restore the video of Beijing hundreds years ago|[Click and Try](https://aistudio.baidu.com/aistudio/projectdetail/1161285)|
|Motion Driving-When "Su Daqiang" sings "unravel" |[Click and Try](https://aistudio.baidu.com/aistudio/projectdetail/1048840)|

## Examples


### Image Translation

<div align='center'>
  <img src='https://user-images.githubusercontent.com/48054808/119464966-d5c1c000-bd75-11eb-9696-9bb75357229f.gif'width='700' height='200'/>
</div>


### Old video restore
<div align='center'>
  <img src='https://user-images.githubusercontent.com/48054808/119469496-fc81f580-bd79-11eb-865a-5e38482b1ae8.gif' width='700'/>
</div>



### Motion driving
<div align='center'>
  <img src='https://user-images.githubusercontent.com/48054808/119469551-0a377b00-bd7a-11eb-9117-e4871c8fb9c0.gif' width='700'>
</div>


### Super resolution

<div align='center'>
  <img src='https://user-images.githubusercontent.com/48054808/119469753-3e12a080-bd7a-11eb-9cde-4fa01b3201ab.png'width='700' height='250'/>
</div>



### Makeup shifter

<div align='center'>
  <img src='https://user-images.githubusercontent.com/48054808/119469834-4ff44380-bd7a-11eb-93b6-05b705dcfbf2.png'width='700' height='250'/>
</div>



### Face cartoonization

<div align='center'>
  <img src='https://user-images.githubusercontent.com/48054808/119469952-6bf7e500-bd7a-11eb-89ad-9a78b10bd4ab.png'width='700' height='250'/>
</div>



### Realistic face cartoonization

<div align='center'>
  <img src='https://user-images.githubusercontent.com/48054808/119470028-7f0ab500-bd7a-11eb-88e9-78a6b9e2e319.png'width='700' height='250'/>
</div>



### Photo animation

<div align='center'>
  <img src='https://user-images.githubusercontent.com/48054808/119470099-9184ee80-bd7a-11eb-8b12-c9400fe01266.png'width='700' height='250'/>
</div>



### Lip-syncing

<div align='center'>
  <img src='https://user-images.githubusercontent.com/48054808/119470166-a6618200-bd7a-11eb-9f98-58052ce21b14.gif'width='700'>
</div>



## Changelog

- v0.1.0 (2020.11.02)
  - Release first version, supported models include Pixel2Pixel, CycleGAN, PSGAN. Supported applications include video frame interpolation, super resolution, colorize images and videos, image animation.
  - Modular design and friendly interface.

## Community

Scan OR Code below to join [PaddleGAN QQ Group：1058398620], you can get offical technical support  here and communicate with other developers/friends. Look forward to your participation!

<div align='center'>
  <img src='./docs/imgs/qq.png'width='250' height='300'/>
</div>

### PaddleGAN Special Interest Group（SIG）

It was first proposed and used by [ACM（Association for Computing Machinery)](https://en.wikipedia.org/wiki/Association_for_Computing_Machinery) in 1961. Top International open source organizations including [Kubernates](https://kubernetes.io/) all adopt the form of SIGs, so that members with the same specific interests can share, learn knowledge and develop projects. These members do not need to be in the same country/region or the same organization, as long as they are like-minded, they can all study, work, and play together with the same goals~

PaddleGAN SIG is such a developer organization that brings together people who interested in GAN. There are frontline developers of PaddlePaddle, senior engineers from the world's top 500, and students from top universities at home and abroad.

We are continuing to recruit developers interested and capable to join us building this project and explore more useful and interesting applications together.

SIG contributions:

- [zhen8838](https://github.com/zhen8838): contributed to AnimeGANv2.
- [Jay9z](https://github.com/Jay9z): contributed to DCGAN and updated install docs, etc.
- [HighCWu](https://github.com/HighCWu): contributed to c-DCGAN and WGAN. Support to use `paddle.vision.datasets`.
- [hao-qiang](https://github.com/hao-qiang) & [ minivision-ai ](https://github.com/minivision-ai): contributed to the photo2cartoon project.


## Contributing

Contributions and suggestions are highly welcomed. Most contributions require you to agree to a [Contributor License Agreement (CLA)](https://cla-assistant.io/PaddlePaddle/PaddleGAN) declaring.
When you submit a pull request, a CLA-bot will automatically determine whether you need to provide a CLA. Simply follow the instructions provided by the bot. You will only need to do this once across all repos using our CLA.
For more, please reference [contribution guidelines](docs/en_US/contribute.md).

## License
PaddleGAN is released under the [Apache 2.0 license](LICENSE).
