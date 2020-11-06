
English | [简体中文](./README.md)

# PaddleGAN

PaddleGAN provides developers with high-performance implementation of classic and SOTA Generative Adversarial Networks, and support developers to quickly build, train and deploy GANs for academic, entertainment and industrial usage.

GAN-Generative Adversarial Network, was praised by "the Father of Convolutional Networks"  **Yann LeCun (Yang Likun)**  as **[One of the most interesting ideas in the field of computer science in the past decade]**. It's one the research area in deep learning that AI researchers are most concerned about.

<div align='center'>
  <img src='./docs/imgs/ppgan.jpg'>
</div>

[![License](https://img.shields.io/badge/license-Apache%202-red.svg)](LICENSE)![python version](https://img.shields.io/badge/python-3.6+-orange.svg)

## Quick Start

* Please refer [install](./docs/en_US/install.md) to ensure you sucessfully installed PaddlePaddle and PaddleGAN.

* Get started through ppgan.app interface:

 ```python
 from ppgan.apps import RealSRPredictor
 sr = RealSRPredictor()
 sr.run("docs/imgs/monarch.png")
 ```

* More tutorials:
  - [Data preparation](./docs/en_US/data_prepare.md)
  - [Training/Evaluating/Testing basic usage](./docs/zh_CN/get_started.md)

## Model Tutorial

* [Pixel2Pixel](./docs/en_US/tutorials/pix2pix_cyclegan.md)
* [CycleGAN](./docs/en_US/tutorials/pix2pix_cyclegan.md)
* [PSGAN](./docs/en_US/tutorials/psgan.md)
* [First Order Motion Model](./docs/en_US/tutorials/motion_driving.md)

## Composite Application

* [Video restore](./docs/zh_CN/tutorials/video_restore.md)

## Examples

### Image Translation

<div align='center'>
  <img src='./docs/imgs/horse2zebra.gif'width='700' height='200'/>
</div>

### Old video restore
<div align='center'>
  <img src='./docs/imgs/color_sr_peking.gif' width='700'/>
</div>


### Motion driving
<div align='center'>
  <img src='./docs/imgs/first_order.gif' width='700'>
</div>

### Super resolution

<div align='center'>
  <img src='./docs/imgs/sr_demo.png'width='700' height='250'/>
</div>


### Makeup shifter

<div align='center'>
  <img src='./docs/imgs/makeup_shifter.png'width='700' height='250'/>
</div>


## Changelog

- v0.1.0 (2020.11.02)
  - Release first version, supported models include Pixel2Pixel, CycleGAN, PSGAN. Supported applications include video frame interpolation, super resolution, colorize images and videos, image animation.
  - Modular design and friendly interface.

## PaddleGAN Special Interest Group（SIG）

It was first proposed and used by [ACM（Association for Computing Machinery)](https://en.wikipedia.org/wiki/Association_for_Computing_Machinery) in 1961. Top International open source organizations including [Kubernates](https://kubernetes.io/) all adopt the form of SIGs, so that members with the same specific interests can share, learn knowledge and develop projects. These members do not need to be in the same country/region or the same organization, as long as they are like-minded, they can all study, work, and play together with the same goals~

PaddleGAN SIG is such a developer organization that brings together people who interested in GAN. There are frontline developers of PaddlePaddle, senior engineers from the world's top 500, and students from top universities at home and abroad.

We are continuing to recruit developers interested and capable to join us building this project and explore more useful and interesting applications together.

[PaddleGAN QQ Group：1058398620]

<div align='center'>
  <img src='./docs/imgs/qq.png'width='250' height='300'/>
</div>



## Contributing

Contributions and suggestions are highly welcomed. Most contributions require you to agree to a [Contributor License Agreement (CLA)](https://cla-assistant.io/PaddlePaddle/PaddleGAN) declaring.
When you submit a pull request, a CLA-bot will automatically determine whether you need to provide a CLA. Simply follow the instructions provided by the bot. You will only need to do this once across all repos using our CLA.
For more, please reference [contribution guidelines](docs/en_US/contribute.md).

## License
PaddleGAN is released under the [Apache 2.0 license](LICENSE).
