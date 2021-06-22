## Install PaddleGAN

### requirements

* PaddlePaddle >= 2.1.0
* Python >= 3.6
* CUDA >= 10.1

### 1. Install PaddlePaddle
```
pip install -U paddlepaddle-gpu
```

Note: command above will install paddle with cuda10.2, if your installed cuda is different, please visit home page of [paddlepaddle](https://www.paddlepaddle.org.cn/install/quick) for more help.

### 2. Install paddleGAN

#### 2.1 Install through pip

```
# only support Python3
python3 -m pip install --upgrade ppgan
```

Download the examples and configuration files via cloning the source code:

```
git clone https://github.com/PaddlePaddle/PaddleGAN
cd PaddleGAN
```

#### 2.2 Install through source code

```
git clone https://github.com/PaddlePaddle/PaddleGAN
cd PaddleGAN
pip install -v -e .  # or "python setup.py develop"
```

### 4. Installation of other tools that may be used

#### 4.1 ffmpeg

If you need to use ppgan to handle video-related tasks, you need to install ffmpeg. It is recommended that you use [conda](https://docs.conda.io/en/latest/miniconda.html) to install:

```
conda install x264=='1!152.20180717' ffmpeg=4.0.2 -c conda-forge
```

#### 4.2 Visual DL
If you want to use [PaddlePaddle VisualDL](https://github.com/PaddlePaddle/VisualDL) to monitor the training process, Please install `VisualDL`(For more detail refer [here](./get_started.md)):

```
python -m pip install visualdl -i https://mirror.baidu.com/pypi/simple
```
