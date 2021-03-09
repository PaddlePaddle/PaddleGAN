## 安装PaddleGAN

### 要求

* PaddlePaddle >= 2.0.0
* Python >= 3.6
* CUDA >= 9.0

### 1. 安装PaddlePaddle
```
pip install -U paddlepaddle-gpu==2.0.0
```

上面命令会默认安装cuda10.2的包，如果想安装其他cuda版本的包或者其他的系统，请参考[paddlepaddle官网安装教程](https://www.paddlepaddle.org.cn/install/quick)

### 2. 安装PaddleGAN

##### 2.1 通过Pip安裝
```
# only support Python3
python3 -m pip install --upgrade ppgan
```

下载示例和配置文件:

```
git clone https://github.com/PaddlePaddle/PaddleGAN
cd PaddleGAN
```

##### 2.2通过源码安装

```
git clone https://github.com/PaddlePaddle/PaddleGAN
cd PaddleGAN

pip install -v -e .  # or "python setup.py develop"
```

按照上述方法安装成功后，本地的修改也会自动同步到ppgan中


### 4. 其他可能用到的工具安装

#### 4.1 ffmpeg

如果需要使用ppgan处理视频相关的任务，则需要安装ffmpeg。这里推荐您使用[conda](https://docs.conda.io/en/latest/miniconda.html)安装：

```
conda install x264=='1!152.20180717' ffmpeg=4.0.2 -c conda-forge
```

#### 4.2 Visual DL

如果需要使用[飞桨VisualDL](https://github.com/PaddlePaddle/VisualDL)对训练过程进行可视化监控，请安装`VisualDL`(使用方法请参考[这里](./get_started.md)):

```
python -m pip install visualdl -i https://mirror.baidu.com/pypi/simple
```
