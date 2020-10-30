## Install PaddleGAN

### requirements

* PaddlePaddle >= 2.0.0-rc
* Python >= 3.5+
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


### 2. Install through pip

```
# only support Python3
python3 -m pip install --upgrade ppgan
```

Download the examples and configuration files via cloning the source code:

```
git clone https://github.com/PaddlePaddle/PaddleGAN
cd PaddleGAN
```

### 3. Install through source code

```
git clone https://github.com/PaddlePaddle/PaddleGAN
cd PaddleGAN
pip install -v -e .  # or "python setup.py develop"
```
