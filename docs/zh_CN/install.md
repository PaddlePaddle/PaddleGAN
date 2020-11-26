## 安装PaddleGAN

### 要求

* PaddlePaddle >= 2.0.0-rc
* Python >= 3.5+
* CUDA >= 9.0

### 1. 安装PaddlePaddle
```
pip install -U paddlepaddle-gpu==2.0.0rc0
```

上面命令会默认安装cuda10.2的包，如果想安装其他cuda版本的包，可以参考下面的表格。
<table class="docutils"><tbody><th width="80"> CUDA </th><th valign="bottom" align="left" width="100">python3.8</th><th valign="bottom" align="left" width="100">python3.7</th><th valign="bottom" align="left" width="100">python3.6</th> <tr><td align="left">10.1</td><td align="left"><details><summary> install </summary><pre><code>python -m pip install https://paddle-wheel.bj.bcebos.com/2.0.0-rc0-gpu-cuda10.1-cudnn7-mkl_gcc8.2%2Fpaddlepaddle_gpu-2.0.0rc0.post101-cp38-cp38-linux_x86_64.whl
</code></pre> </details> </td> <td align="left"><details><summary> install </summary><pre><code>python -m pip install https://paddle-wheel.bj.bcebos.com/2.0.0-rc0-gpu-cuda10.1-cudnn7-mkl_gcc8.2%2Fpaddlepaddle_gpu-2.0.0rc0.post101-cp37-cp37m-linux_x86_64.whl
</code></pre> </details> </td> <td align="left"><details><summary> install </summary><pre><code>python -m pip install https://paddle-wheel.bj.bcebos.com/2.0.0-rc0-gpu-cuda10.1-cudnn7-mkl_gcc8.2%2Fpaddlepaddle_gpu-2.0.0rc0.post101-cp36-cp36m-linux_x86_64.whl
</code></pre> </details> </td> <td align="left"> </td> </tr> <tr><td align="left">10.0</td><td align="left"><details><summary> install </summary><pre><code>python -m pip install https://paddle-wheel.bj.bcebos.com/2.0.0-rc0-gpu-cuda10-cudnn7-mkl%2Fpaddlepaddle_gpu-2.0.0rc0.post100-cp38-cp38-linux_x86_64.whl
</code></pre> </details> </td> <td align="left"><details><summary> install </summary><pre><code>python -m pip install https://paddle-wheel.bj.bcebos.com/2.0.0-rc0-gpu-cuda10-cudnn7-mkl%2Fpaddlepaddle_gpu-2.0.0rc0.post100-cp37-cp37m-linux_x86_64.whl
</code></pre> </details> </td> <td align="left"><details><summary> install </summary><pre><code>python -m pip install https://paddle-wheel.bj.bcebos.com/2.0.0-rc0-gpu-cuda10-cudnn7-mkl%2Fpaddlepaddle_gpu-2.0.0rc0.post100-cp36-cp36m-linux_x86_64.whl
</code></pre> </details> </td> <td align="left"> </td> </tr> <tr><td align="left">9.0</td><td align="left"><details><summary> install </summary><pre><code>python -m pip install https://paddle-wheel.bj.bcebos.com/2.0.0-rc0-gpu-cuda9-cudnn7-mkl%2Fpaddlepaddle_gpu-2.0.0rc0.post90-cp38-cp38-linux_x86_64.whl
</code></pre> </details> </td> <td align="left"><details><summary> install </summary><pre><code>python -m pip install https://paddle-wheel.bj.bcebos.com/2.0.0-rc0-gpu-cuda9-cudnn7-mkl%2Fpaddlepaddle_gpu-2.0.0rc0.post90-cp37-cp37m-linux_x86_64.whl
</code></pre> </details> </td> <td align="left"><details><summary> install </summary><pre><code>python -m pip install https://paddle-wheel.bj.bcebos.com/2.0.0-rc0-gpu-cuda9-cudnn7-mkl%2Fpaddlepaddle_gpu-2.0.0rc0.post90-cp36-cp36m-linux_x86_64.whl
</code></pre> </details> </td> </tr></tbody></table>

支持更多系统的安装教程请前往[paddlepaddle官网](https://www.paddlepaddle.org.cn/install/quick)

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
