# 安装文档
本文档包含了如何安装PaddleGAN以及相关依赖，更多产品简介请参考[README](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/README_cn.md)。

## 环境依赖

- PaddlePaddle >= 2.1.0
- Python >= 3.6
- CUDA >= 10.1


## 安装PaddlePaddle

```

# CUDA10.1
python -m pip install paddlepaddle-gpu==2.1.0.post101 -f https://mirror.baidu.com/pypi/simple

# CPU
python -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple

```

更多安装方式例如conda或源码编译安装方法，请参考[PaddlePaddle安装文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/index_cn.html)。

请确保您的PaddlePaddle安装成功并且版本不低于需求版本。使用以下命令进行验证。

```
# 在您的Python解释器中确认PaddlePaddle安装成功
>>> import paddle
>>> paddle.utils.run_check()

# 确认PaddlePaddle版本
python -c "import paddle; print(paddle.__version__)"
```

## 安装PaddleGAN

### 通过PIP安裝（只支持Python3）

* 安装：
```
python3 -m pip install --upgrade ppgan
```
* 下载示例和配置文件:

```
git clone https://github.com/PaddlePaddle/PaddleGAN
cd PaddleGAN
```
### 通过源码安装

```
git clone https://github.com/PaddlePaddle/PaddleGAN
cd PaddleGAN

pip install -v -e .  # or "python setup.py develop"

# 安装其他依赖
pip install -r requirements.txt
```
## 其他第三方工具安装

* 涉及视频的任务都需安装**ffmpeg**，这里推荐使用[conda](https://docs.conda.io/en/latest/miniconda.html)安装：

```
conda install x264=='1!152.20180717' ffmpeg=4.0.2 -c conda-forge
```

* 如需使用可视化工具监控训练过程，请安装[飞桨VisualDL](https://github.com/PaddlePaddle/VisualDL)：
```
python -m pip install visualdl -i https://mirror.baidu.com/pypi/simple
```
*注意：VisualDL目前只维护Python3以上的安装版本
