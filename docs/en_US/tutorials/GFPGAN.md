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

```
[09/15 13:14:57] ppgan.engine.trainer INFO: Iter: 210000/800000 lr: 2.000e-03 l_g_pix: 0.012 l_p_8: 0.000 l_p_16: 0.000 l_p_32: 0.000 l_p_64: 0.000 l_p_128: 0.000 l_p_256: 0.000 l_p_512: 0.000 l_g_percep: 9.977 l_g_style: 1.725 l_g_gan: 0.301 l_g_gan_left_eye: 0.731 l_g_gan_right_eye: 0.848 l_g_gan_mouth: 0.898 l_g_comp_style_loss: 1.014 l_identity: 0.502 l_d: 0.517 real_score: 0.574 fake_score: -2.958 l_d_left_eye: 1.450 l_d_right_eye: 1.382 l_d_mouth: 1.266 l_d_r1: 4.474 batch_cost: 1.12074 sec reader_cost: 0.00216 sec ips: 2.67681 images/s eta: 7 days, 15:40:36
[09/15 13:14:58] ppgan.engine.trainer INFO: Test iter: [0/252]
[09/15 13:15:00] ppgan.engine.trainer INFO: Test iter: [4/252]
[09/15 13:15:01] ppgan.engine.trainer INFO: Test iter: [8/252]
[09/15 13:15:02] ppgan.engine.trainer INFO: Test iter: [12/252]

[09/15 13:15:58] ppgan.engine.trainer INFO: Test iter: [248/252]
[09/15 13:15:59] ppgan.engine.trainer INFO: Metric psnr: 65.0461
[09/15 13:16:13] ppgan.engine.trainer INFO: Metric fid: 36.8068
[09/15 13:16:14] ppgan.engine.trainer INFO: Metric LPIPS: 0.3817
```

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


Please modify the dataroot parameters of dataset train and test in the configs/gfpgan_1024_ffhq.yaml configuration file to your training set and test set path.


### 2.2 Model preparation

**Model parameter file and training log download address:**

https://paddlegan.bj.bcebos.com/models/GFPGAN.pdparams

Download the model parameters and test images from the link and put them in the data/ folder in the project root directory. The specific file structure is as follows:

the params is a dict(one type in python),and could be load by paddlepaddle. It contains key (net_g,net_g_ema),you can use any of one to inference

## 3、Start using

### 3.1 model training

Enter the following code in the console to start training：

 ```shell
 python tools/main.py -c configs/gfpgan_1024_ffhq.yaml
 ```

The model only supports single-card training.

Model training needs to use paddle2.3 and above, and wait for paddle to implement the second-order operator related functions of elementwise_pow. The paddle2.2.2 version can run normally, but the model cannot be successfully trained because some loss functions will calculate the wrong gradient. . If an error is reported during training, training is not supported for the time being. You can skip the training part and directly use the provided model parameters for testing. Model evaluation and testing can use paddle2.2.2 and above.



### 3.2 Model evaluation

When evaluating the model, enter the following code in the console, using the downloaded model parameters mentioned above:

 ```shell
python tools/main.py -c configs/gfpgan_1024_ffhq.yaml --load GFPGAN.pdparams --evaluate-only
 ```

If you want to test on your own provided model, please modify the path after --load .



### 3.3 Model prediction

#### 3.3.1 Export model

After training, you need to use ``tools/export_model.py`` to extract the weights of the generator from the trained model (including the generator only)
Enter the following command to extract the model of the generator:

```bash
python -u tools/export_model.py --config-file configs/gfpgan_1024_ffhq.yaml \
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

### 4.1 Export the inference model

```bash
python -u tools/export_model.py --config-file configs/gfpgan_1024_ffhq.yaml \
    --load GFPGAN.pdparams \
    --inputs_size 1,3,512,512
```

You can also modify the parameters after --load to the model parameter file you want to test.



### 4.2 Inference with a prediction engine

```bash
%cd /home/aistudio/work/PaddleGAN
# %env PYTHONPATH=.:$PYTHONPATH
# %env CUDA_VISIBLE_DEVICES=0
!python -u tools/inference.py --config-file configs/gfpgan_1024_ffhq.yaml \
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
@misc{2021GAN,
      title={GAN Prior Embedded Network for Blind Face Restoration in the Wild},
      author={ Yang, T.  and  Ren, P.  and  Xie, X.  and  Zhang, L. },
      year={2021},
      archivePrefix={CVPR},
      primaryClass={cs.CV}
}
```
