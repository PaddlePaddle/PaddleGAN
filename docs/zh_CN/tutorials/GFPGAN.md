## GFPGAN 盲脸复原模型


## 1、介绍
GFP-GAN利用丰富和多样化的先验封装在预先训练的面部GAN用于盲人面部恢复。
### GFPGAN的整体结构:

![image](https://user-images.githubusercontent.com/73787862/191736718-72f5aa09-d7a9-490b-b1f8-b609208d4654.png)

GFP-GAN由降解去除物组成
模块(U-Net)和预先训练的面部GAN(如StyleGAN2)作为先验。他们之间有隐藏的密码
映射和几个通道分割空间特征变换(CS-SFT)层。

通过处理特征，它在保持高保真度的同时实现了真实的结果。

要了解更详细的模型介绍，并参考回购，您可以查看以下AI Studio项目
[基于PaddleGAN复现GFPGAN](https://aistudio.baidu.com/aistudio/projectdetail/4421649)

在这个实验中，我们训练
我们的模型和Adam优化器共进行了210k次迭代。

GFPGAN的回收实验结果如下:


Model | LPIPS | FID | PSNR
--- |:---:|:---:|:---:|
GFPGAN | 0.3817 | 36.8068 | 65.0461

## 2、准备工作

### 2.1 数据集准备

GFPGAN模型训练集是经典的FFHQ人脸数据集，
总共有7万张高分辨率1024 x 1024的人脸图片，
测试集为CELEBA-HQ数据集，共有2000张高分辨率人脸图片。生成方式与训练时相同。
For details, please refer to **Dataset URL:** [FFHQ](https://github.com/NVlabs/ffhq-dataset), [CELEBA-HQ](https://github.com/tkarras/progressive_growing_of_gans).
The specific download links are given below:

**原始数据集地址:**

**FFHQ ：**           https://drive.google.com/drive/folders/1tZUcXDBeOibC6jcMCtgRRz67pzrAHeHL?usp=drive_open

**CELEBA-HQ：** https://drive.google.com/drive/folders/0B4qLcYyJmiz0TXY1NG02bzZVRGs?resourcekey=0-arAVTUfW9KRhN-irJchVKQ&usp=sharing

数据集结构如下

```
|-- data/GFPGAN
    |-- train
        |-- 00000.png
        |-- 00001.png
        |-- ......
        |-- 00999.png
        |-- ......
        |-- 69999.png
	|-- lq
		|-- 2000张jpg图片
    |-- gt  
        |-- 2000张jpg图片
```

请在configs/gfpgan_ffhq1024. data中修改数据集train和test的dataroot参数。Yaml配置文件到您的训练集和测试集路径。

### 2.2 模型准备
**模型参数文件和训练日志下载地址:**

https://paddlegan.bj.bcebos.com/models/GFPGAN.pdparams

从链接下载模型参数和测试图像，并将它们放在项目根目录中的data/文件夹中。具体文件结构如下:

params是一个dict(python中的一种类型)，可以通过paddlepaddle加载。它包含key (net_g,net_g_ema)，您可以使用其中任何一个来进行推断

## 3、开始使用
模型训练

在控制台中输入以下代码开始训练:

 ```bash
 python tools/main.py -c configs/gfpgan_ffhq1024.yaml
 ```

该模型支持单卡训练和多卡训练。
也可以使用如下命令进行多卡训练

```bash
!CUDA_VISIBLE_DEVICES=0,1,2,3
!python -m paddle.distributed.launch tools/main.py \
        --config-file configs/gpfgan_ffhq1024.yaml
```

模型训练需要使用paddle2.3及以上版本，等待paddle实现elementwise_pow的二阶算子相关函数。paddle2.2.2版本可以正常运行，但由于某些损失函数会计算出错误的梯度，无法成功训练模型。如果在培训过程中报错，则暂时不支持培训。您可以跳过训练部分，直接使用提供的模型参数进行测试。模型评估和测试可以使用paddle2.2.2及以上版本。

### 3.2 模型评估

当评估模型时，在控制台中输入以下代码，使用上面提到的下载的模型参数:

 ```shell
python tools/main.py -c configs/gfpgan_ffhq1024.yaml --load GFPGAN.pdparams --evaluate-only
 ```

当评估模型时，在控制台中输入以下代码，使用下载的模型。如果您想在您自己提供的模型上进行测试，请修改之后的路径 --load .



### 3.3 模型预测

#### 3.3.1 导出模型

在训练之后，您需要使用' ' tools/export_model.py ' '从训练的模型中提取生成器的权重(仅包括生成器)
输入以下命令提取生成器的模型:

```bash
python -u tools/export_model.py --config-file configs/gfpgan_ffhq1024.yaml \
    --load GFPGAN.pdparams \
    --inputs_size 1,3,512,512
```


#### 3.3.2 加载一张图片

你可以使用我们在ppgan/faceutils/face_enhancement/gfpgan_enhance.py中的工具来快速推断一张图片

```python
%env PYTHONPATH=.:$PYTHONPATH
%env CUDA_VISIBLE_DEVICES=0
import paddle
import cv2
import numpy as np
import sys
from ppgan.faceutils.face_enhancement.gfpgan_enhance import gfp_FaceEnhancement
# 图片路径可以用自己的
img_path='test/2.png'
img = cv2.imread(img_path, cv2.IMREAD_COLOR)
# 这是原来的模糊图片
cv2.imwrite('test/outlq.png',img)
img=np.array(img).astype('float32')
faceenhancer = gfp_FaceEnhancement()
img = faceenhancer.enhance_from_image(img)
# 这是生成的清晰图片
cv2.imwrite('test/out_gfpgan.png',img)
```

![image](https://user-images.githubusercontent.com/73787862/191741112-b813a02c-6b19-4591-b80d-0bf5ce8ad07e.png)
![image](https://user-images.githubusercontent.com/73787862/191741242-1f365048-ba25-450f-8abc-76e74d8786f8.png)




## 4. Tipc

### 4.1 导出推理模型

```bash
python -u tools/export_model.py --config-file configs/gfpgan_ffhq1024.yaml \
    --load GFPGAN.pdparams \
    --inputs_size 1,3,512,512
```

### 4.2 使用paddleInference推理

```bash
%cd /home/aistudio/work/PaddleGAN
# %env PYTHONPATH=.:$PYTHONPATH
# %env CUDA_VISIBLE_DEVICES=0
!python -u tools/inference.py --config-file configs/gfpgan_ffhq1024.yaml \
    --model_path GFPGAN.pdparams \
    --model_type gfpgan \
    --device gpu \
    -o validate=None
```


### 4.3 一键TIPC

调用足部测试基础训练预测函数的' lite_train_lite_infer '模式，执行:

```bash
%cd /home/aistudio/work/PaddleGAN
!bash test_tipc/prepare.sh \
    test_tipc/configs/GFPGAN/train_infer_python.txt \
    lite_train_lite_infer
!bash test_tipc/test_train_inference_python.sh \
    test_tipc/configs/GFPGAN/train_infer_python.txt \
    lite_train_lite_infer
```



## 5、References

```
@InProceedings{wang2021gfpgan,
    author = {Xintao Wang and Yu Li and Honglun Zhang and Ying Shan},
    title = {Towards Real-World Blind Face Restoration with Generative Facial Prior},
    booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2021}
}
```
