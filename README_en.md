English | [简体中文](./README_cn.md)

# PaddleGAN

PaddleGAN is an development kit of Generative Adversarial Network based on PaddlePaddle.

### Image Translation
![](./docs/imgs/A2B.png)
![](./docs/imgs/B2A.png)

### Makeup shifter
![](./docs/imgs/makeup_shifter.png)

### Old video restore
![](./docs/imgs/color_sr_peking.gif)

### Super resolution

![](./docs/imgs/sr_demo.png)

### Motion driving
![](./docs/imgs/first_order.gif)

Features:

- Highly Flexible:

  Components are designed to be modular. Model architectures, as well as data
preprocess pipelines, can be easily customized with simple configuration
changes.

- Rich applications:

  PaddleGAN provides rich of applications, such as image generation, image restore, image colorization, video interpolate, makeup shifter.

## Install

### 1. install paddlepaddle

PaddleGAN work with:
* PaddlePaddle >= 2.0.0-rc
* Python >= 3.5+

```
pip install -U paddlepaddle-gpu
```

### 2. install ppgan

```
python -m pip install 'git+https://github.com/PaddlePaddle/PaddleGAN.git'
```

Or install it from a local clone
```
git clone https://github.com/PaddlePaddle/PaddleGAN
cd PaddleGAN

pip install -v -e .  # or "python setup.py develop"
```

## Data Prepare
Please refer to [data prepare](./docs/data_prepare.md) for dataset preparation.

## Get Start
Please refer [get started](./docs/get_started.md) for the basic usage of PaddleGAN.

## Model tutorial
* [Pixel2Pixel and CycleGAN](./docs/tutorials/pix2pix_cyclegan.md)
* [PSGAN](./docs/tutorials/psgan_en.md)
* [Video restore](./docs/tutorails/video_restore.md)
* [Motion driving](./docs/tutorials/motion_driving_en.md)

## License
PaddleGAN is released under the [Apache 2.0 license](LICENSE).

## Contributing

Contributions and suggestions are highly welcomed. Most contributions require you to agree to a [Contributor License Agreement (CLA)](https://cla-assistant.io/PaddlePaddle/PaddleGAN) declaring.
When you submit a pull request, a CLA-bot will automatically determine whether you need to provide a CLA. Simply follow the instructions provided by the bot. You will only need to do this once across all repos using our CLA.
For more, please reference [contribution guidelines](docs/CONTRIBUTE.md).


## External Projects

External gan projects in the community that base on PaddlePaddle:

+ [PaddleGAN](https://github.com/PaddlePaddle/PaddleGAN)
