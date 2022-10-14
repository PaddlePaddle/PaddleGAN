## GFPGAN Blind Face Restoration Model



## 1、Introduction

GFP-GAN that leverages rich and diverse priors encapsulated in a pretrained face GAN for blind face restoration.
### Overview of GFP-GAN framework:

![image](https://user-images.githubusercontent.com/73787862/191736718-72f5aa09-d7a9-490b-b1f8-b609208d4654.png)

GFP-GAN is comprised of a degradation removal
module (U-Net) and a pretrained face GAN (such as StyleGAN2) as prior. They are bridged by a latent code
mapping and several Channel-Split Spatial Feature Transform (CS-SFT) layers.

By dealing with features, it achieving realistic results while preserving high fidelity.

For a more detailed introduction to the model, and refer to the repo, you can view the following AI Studio project
[https://aistudio.baidu.com/aistudio/projectdetail/4421649](https://aistudio.baidu.com/aistudio/projectdetail/4421649)

In this experiment, We train
our model with Adam optimizer for a total of 210k iterations.

The result of experiments of recovering of GFPGAN as following:

Model | LPIPS | FID | PSNR
--- |:---:|:---:|:---:|
GFPGAN | 0.3817 | 36.8068 | 65.0461

## 2、Ready to work

### 2.1 Dataset Preparation

The GFPGAN model training set is the classic FFHQ face data set,
with a total of 70,000 high-resolution 1024 x 1024 high-resolution face pictures,
and the test set is the CELEBA-HQ data set, with a total of 2,000 high-resolution face pictures. The generation way is the same as that during training.
For details, please refer to **Dataset URL:** [FFHQ](https://github.com/NVlabs/ffhq-dataset), [CELEBA-HQ](https://github.com/tkarras/progressive_growing_of_gans).
The specific download links are given below:

**Original dataset download address:**

**FFHQ ：**           https://drive.google.com/drive/folders/1tZUcXDBeOibC6jcMCtgRRz67pzrAHeHL?usp=drive_open

**CELEBA-HQ：** https://drive.google.com/drive/folders/0B4qLcYyJmiz0TXY1NG02bzZVRGs?resourcekey=0-arAVTUfW9KRhN-irJchVKQ&usp=sharing

The structure of data as following

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


Please modify the dataroot parameters of dataset train and test in the configs/gfpgan_ffhq1024.yaml configuration file to your training set and test set path.


### 2.2 Model preparation

**Model parameter file and training log download address:**

https://paddlegan.bj.bcebos.com/models/GFPGAN.pdparams

Download the model parameters and test images from the link and put them in the data/ folder in the project root directory. The specific file structure is as follows:

the params is a dict(one type in python),and could be load by paddlepaddle. It contains key (net_g,net_g_ema),you can use any of one to inference

## 3、Start using

### 3.1 model training

Enter the following code in the console to start training：

 ```bash
 python tools/main.py -c configs/gfpgan_ffhq1024.yaml
 ```

The model supports single-card training and multi-card training.So you can use this bash to train

 ```bash
!CUDA_VISIBLE_DEVICES=0,1,2,3
!python -m paddle.distributed.launch tools/main.py \
        --config-file configs/gpfgan_ffhq1024.yaml
 ```

Model training needs to use paddle2.3 and above, and wait for paddle to implement the second-order operator related functions of elementwise_pow. The paddle2.2.2 version can run normally, but the model cannot be successfully trained because some loss functions will calculate the wrong gradient. . If an error is reported during training, training is not supported for the time being. You can skip the training part and directly use the provided model parameters for testing. Model evaluation and testing can use paddle2.2.2 and above.



### 3.2 Model evaluation

When evaluating the model, enter the following code in the console, using the downloaded model parameters mentioned above:

 ```shell
python tools/main.py -c configs/gfpgan_ffhq1024.yaml --load GFPGAN.pdparams --evaluate-only
 ```

If you want to test on your own provided model, please modify the path after --load .



### 3.3 Model prediction

#### 3.3.1 Export model

After training, you need to use ``tools/export_model.py`` to extract the weights of the generator from the trained model (including the generator only)
Enter the following command to extract the model of the generator:

```bash
python -u tools/export_model.py --config-file configs/gfpgan_ffhq1024.yaml \
    --load GFPGAN.pdparams \
    --inputs_size 1,3,512,512
```


#### 3.3.2 Process a single image

You can use our tools in ppgan/faceutils/face_enhancement/gfpgan_enhance.py to inferences one picture quickly
```python
%env PYTHONPATH=.:$PYTHONPATH
%env CUDA_VISIBLE_DEVICES=0
import paddle
import cv2
import numpy as np
import sys
from ppgan.faceutils.face_enhancement.gfpgan_enhance import gfp_FaceEnhancement
# you can use your path
img_path='test/2.png'
img = cv2.imread(img_path, cv2.IMREAD_COLOR)
# this is origin picture
cv2.imwrite('test/outlq.png',img)
img=np.array(img).astype('float32')
faceenhancer = gfp_FaceEnhancement()
img = faceenhancer.enhance_from_image(img)
# the result of prediction
cv2.imwrite('test/out_gfpgan.png',img)
```

![image](https://user-images.githubusercontent.com/73787862/191741112-b813a02c-6b19-4591-b80d-0bf5ce8ad07e.png)
![image](https://user-images.githubusercontent.com/73787862/191741242-1f365048-ba25-450f-8abc-76e74d8786f8.png)




## 4. Tipc

### 4.1 Export the inference model

```bash
python -u tools/export_model.py --config-file configs/gfpgan_ffhq1024.yaml \
    --load GFPGAN.pdparams \
    --inputs_size 1,3,512,512
```

You can also modify the parameters after --load to the model parameter file you want to test.



### 4.2 Inference with a prediction engine

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


### 4.3 Call the script to complete the training and push test in two steps

To invoke the `lite_train_lite_infer` mode of the foot test base training prediction function, run:

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
