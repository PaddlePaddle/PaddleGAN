

## Installation

This document contains how to install PaddleGAN and related dependencies. For more product overview, please refer to [README](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/README_en.md).

### Requirements

* PaddlePaddle >= 2.1.0
* Python >= 3.6
* CUDA >= 10.1

### Install PaddlePaddle
```
# CUDA10.1
python -m pip install paddlepaddle-gpu==2.1.0.post101 -f https://mirror.baidu.com/pypi/simple

# CPU
python -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
```

For more installation methods such as conda or source compilation installation methods, please refer to the [PaddlePaddle installation documentation](https://www.paddlepaddle.org.cn/documentation/docs/en/install/index_en.html).

Make sure that your PaddlePaddle is successfully installed in the required or higher version, and then please use the following command to verify.

```
# verify that PaddlePaddle is installed successfully in your Python interpreter
>>> import paddle
>>> paddle.utils.run_check()

# Confirm PaddlePaddle version
python -c "import paddle; print(paddle.__version__)"
```

### Install PaddleGAN

#### 1. Install via PIP (only Python3 is available)

* Install

```
python3 -m pip install --upgrade ppgan
```

* Download the examples and configuration files via cloning the source code:

```
git clone https://github.com/PaddlePaddle/PaddleGAN
cd PaddleGAN
```

#### 2. Install via source code

```
git clone https://github.com/PaddlePaddle/PaddleGAN
cd PaddleGAN

pip install -v -e .  # or "python setup.py develop"

# Install other dependencies
pip install -r requirements.txt
```

### Other Third-Party Tool Installation

#### 1. ffmpeg

All tasks involving video require `ffmpeg` to be installed, here we recommend using conda

```
conda install x264=='1!152.20180717' ffmpeg=4.0.2 -c conda-forge
```

#### 2. VisualDL
If you want to use [PaddlePaddle VisualDL](https://github.com/PaddlePaddle/VisualDL) to visualize the training process, Please install `VisualDL`(For more detail refer [here](./get_started.md)):

```
python -m pip install visualdl -i https://mirror.baidu.com/pypi/simple
```

*Note: Only versions installed under Python 3 or higher are maintained by VisualDL officially.
