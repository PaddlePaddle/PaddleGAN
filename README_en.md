<div align='center'>
  <img src='./docs/imgs/ppgan.jpg'>
</div>

English | [简体中文](./README.md)

# PaddleGAN

PaddleGAN is an development kit of Generative Adversarial Network based on PaddlePaddle.

### Image Translation
<div align='center'>
  <img src='./docs/imgs/horse2zebra.gif'>
</div>


### Makeup shifter
<div align='center'>
  <img src='./docs/imgs/makeup_shifter.png'>
</div>


### Old video restore
<div align='center'>
  <img src='./docs/imgs/color_sr_peking.gif'>
</div>

### Super resolution

<div align='center'>
  <img src='./docs/imgs/sr_demo.png'>
</div>

### Motion driving
<div align='center'>
  <img src='./docs/imgs/first_order.gif'>
</div>

Features:

- Highly Flexible:

  Components are designed to be modular. Model architectures, as well as data
preprocess pipelines, can be easily customized with simple configuration
changes.

- Rich applications:

  PaddleGAN provides rich of applications, such as image generation, image restore, image colorization, video interpolate, makeup shifter.

## Install

Please refer to [install](./docs/en_US/install.md).

## Quick Start

Get started through ppgan.app interface:

 ```python
 from ppgan.apps import RealSRPredictor
 sr = RealSRPredictor()
 sr.run("docs/imgs/monarch.png")
 ```

More tutorials:

- [Data preparation](./docs/en_US/data_prepare.md)
- [Traning/Evaluating/Testing basic usage](./docs/zh_CN/get_started.md)

## Model Tutorial

* [Pixel2Pixel](./docs/en_US/tutorials/pix2pix_cyclegan.md)
* [CycleGAN](./docs/en_US/tutorials/pix2pix_cyclegan.md)
* [PSGAN](./docs/en_US/tutorials/psgan.md)
* [First Order Motion Model](./docs/en_US/tutorials/motion_driving.md)
* [Video restore](./docs/zh_CN/tutorials/video_restore.md)


## Changelog

- v0.1.0 (2020.11.02)
  - Realse first version, supported models include Pixel2Pixel, CycleGAN, PSGAN. Supported applications include video frame interpolation, super resolution, colorize images and videos, image animation.
  - Modular design and friendly interface.

## Contributing

Contributions and suggestions are highly welcomed. Most contributions require you to agree to a [Contributor License Agreement (CLA)](https://cla-assistant.io/PaddlePaddle/PaddleGAN) declaring.
When you submit a pull request, a CLA-bot will automatically determine whether you need to provide a CLA. Simply follow the instructions provided by the bot. You will only need to do this once across all repos using our CLA.
For more, please reference [contribution guidelines](docs/en_US/contribute.md).

## License
PaddleGAN is released under the [Apache 2.0 license](LICENSE).
