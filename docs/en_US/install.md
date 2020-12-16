## Install PaddleGAN

### requirements

* PaddlePaddle >= 2.0.0-rc
* Python >= 3.6
* CUDA >= 9.0

### 1. Install PaddlePaddle
```
pip install -U paddlepaddle-gpu==2.0.0rc0
```

Note: command above will install paddle with cuda10.2，if your installed cuda is different, you can choose an proper version to install from table below.

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

Visit home page of [paddlepaddle](https://www.paddlepaddle.org.cn/install/quick) for support of other systems, such as Windows10.

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
