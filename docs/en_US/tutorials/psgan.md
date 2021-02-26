# PSGAN

## 1. PSGAN introduction

This paper is to address the makeup transfer task, which aims to transfer the makeup from a reference image to a source image. Existing methods have achieved promising progress in constrained scenarios, but transferring between images with large pose and expression differences is still challenging. To address these issues, we propose Pose and expression robust Spatial-aware GAN ([PSGAN](https://arxiv.org/abs/1909.06956)). It first utilizes Makeup Distill Network to disentangle the makeup of the reference image as two spatial-aware makeup matrices. Then, Attentive Makeup Morphing module is introduced to specify how the makeup of a pixel in the source image is morphed from the reference image. With the makeup matrices and the source image, Makeup Apply Network is used to perform makeup transfer.

<div align="center">
  <img src="../../imgs/psgan_arc.png" width="800"/>
</div>

## 2. How to use
### 2.1 Test
Pretrained model can be downloaded under following link: [psgan_weight](https://paddlegan.bj.bcebos.com/models/psgan_weight.pdparams)

Running the following command to complete the makeup transfer task. It will geneate the transfered image in the current path when the program running sucessfully.

```
python tools/psgan_infer.py \  
  --config-file configs/makeup.yaml \
  --model_path /your/model/path \
  --source_path  docs/imgs/ps_source.png  \
  --reference_dir docs/imgs/ref \
  --evaluate-only
```
**params:**
- config-file: PSGAN network configuration file, yaml format
- model_path: Saved model weight path
- source_path: Full path of the non-makeup image file, including the image file name
- reference_dir: Path of the make_up iamge file, don't including the image file name

### 2.2 Training
1. Downloading the original makeup transfer [data](https://pan.baidu.com/s/1ZF-DN9PvbBteOSfQodWnyw)(Password:rtdd) to the PaddleGAN folder, and uncompress it.
2. Downloading the landmarks [data](https://paddlegan.bj.bcebos.com/landmarks.tar), and uncompress it
3. Runnint the following command to substitute files:
```
rm -rf MT-Dataset/landmarks/makeup && mv landmarks/makeup MT-Dataset/landmarks/
rm -rf MT-Dataset/landmarks/non-makeup && mv landmarks/non-makeup MT-Dataset/landmarks/
cp landmarks/train_makeup.txt MT-Dataset/train_makeup.txt
cp landmarks/train_non-makeup.txt MT-Dataset/train_non-makeup.txt
```

The final data directory should be looked like:

```
data/MT-Dataset
├── images
│   ├── makeup
│   └── non-makeup
├── landmarks
│   ├── makeup
│   └── non-makeup
├── train_makeup.txt
├── train_non-makeup.txt
├── segs
│   ├── makeup
│   └── non-makeup
```

2. `python tools/main.py --config-file configs/makeup.yaml`

The training log looks like:
```
[10/29 05:39:40] ppgan.engine.trainer INFO: Epoch: 0, iters: 0 lr: 0.000200 D_A: 0.448 G_A: 0.973 rec: 1.258 idt: 0.624 D_B: 0.436 G_B: 0.889 G_A_his: 0.402 G_B_his: 0.472 G_bg_consis: 0.030 A_vgg: 0.027 B_vgg: 0.040 reader cost: 2.45463s batch cost: 4.20075s
[10/29 05:40:00] ppgan.engine.trainer INFO: Epoch: 0, iters: 10 lr: 0.000200 D_A: 0.200 G_A: 0.488 rec: 0.954 idt: 0.539 D_B: 0.179 G_B: 0.767 G_A_his: 0.224 G_B_his: 0.266 G_bg_consis: 0.033 A_vgg: 0.019 B_vgg: 0.026 reader cost: 0.55506s batch cost: 1.95968s
[10/29 05:40:22] ppgan.engine.trainer INFO: Epoch: 0, iters: 20 lr: 0.000200 D_A: 0.340 G_A: 0.339 rec: 1.293 idt: 0.698 D_B: 0.124 G_B: 0.174 G_A_his: 0.302 G_B_his: 0.233 G_bg_consis: 0.061 A_vgg: 0.032 B_vgg: 0.045 reader cost: 0.74937s batch cost: 2.13529s
[10/29 05:40:42] ppgan.engine.trainer INFO: Epoch: 0, iters: 30 lr: 0.000200 D_A: 0.238 G_A: 0.276 rec: 0.907 idt: 0.449 D_B: 0.324 G_B: 0.292 G_A_his: 0.263 G_B_his: 0.380 G_bg_consis: 0.029 A_vgg: 0.040 B_vgg: 0.049 reader cost: 0.69248s batch cost: 2.06999s
[10/29 05:41:03] ppgan.engine.trainer INFO: Epoch: 0, iters: 40 lr: 0.000200 D_A: 0.236 G_A: 0.111 rec: 0.865 idt: 0.470 D_B: 0.237 G_B: 0.465 G_A_his: 0.289 G_B_his: 0.211 G_bg_consis: 0.021 A_vgg: 0.042 B_vgg: 0.049 reader cost: 0.65904s batch cost: 2.07197s
[10/29 05:41:23] ppgan.engine.trainer INFO: Epoch: 0, iters: 50 lr: 0.000200 D_A: 0.341 G_A: 0.073 rec: 0.698 idt: 0.424 D_B: 0.153 G_B: 0.731 G_A_his: 0.198 G_B_his: 0.180 G_bg_consis: 0.019 A_vgg: 0.032 B_vgg: 0.047 reader cost: 0.52772s batch cost: 1.92949s
[10/29 05:41:43] ppgan.engine.trainer INFO: Epoch: 0, iters: 60 lr: 0.000200 D_A: 0.267 G_A: 0.475 rec: 0.843 idt: 0.462 D_B: 0.266 G_B: 0.534 G_A_his: 0.259 G_B_his: 0.219 G_bg_consis: 0.024 A_vgg: 0.031 B_vgg: 0.041 reader cost: 0.58212s batch cost: 2.02212s
[10/29 05:42:03] ppgan.engine.trainer INFO: Epoch: 0, iters: 70 lr: 0.000200 D_A: 0.116 G_A: 0.298 rec: 0.983 idt: 0.543 D_B: 0.097 G_B: 0.233 G_A_his: 0.210 G_B_his: 0.169 G_bg_consis: 0.046 A_vgg: 0.028 B_vgg: 0.034 reader cost: 0.56367s batch cost: 1.97049s
[10/29 05:42:23] ppgan.engine.trainer INFO: Epoch: 0, iters: 80 lr: 0.000200 D_A: 0.325 G_A: 0.339 rec: 0.744 idt: 0.417 D_B: 0.292 G_B: 0.310 G_A_his: 0.189 G_B_his: 0.206 G_bg_consis: 0.016 A_vgg: 0.029 B_vgg: 0.034 reader cost: 0.60760s batch cost: 2.04126s
[10/29 05:42:43] ppgan.engine.trainer INFO: Epoch: 0, iters: 90 lr: 0.000200 D_A: 0.177 G_A: 0.308 rec: 0.970 idt: 0.494 D_B: 0.199 G_B: 0.813 G_A_his: 0.116 G_B_his: 0.153 G_bg_consis: 0.036 A_vgg: 0.019 B_vgg: 0.042 reader cost: 0.62142s batch cost: 1.96606s
[10/29 05:43:03] ppgan.engine.trainer INFO: Epoch: 0, iters: 100 lr: 0.000200 D_A: 0.178 G_A: 0.382 rec: 1.358 idt: 0.607 D_B: 0.265 G_B: 0.405 G_A_his: 0.086 G_B_his: 0.161 G_bg_consis: 0.060 A_vgg: 0.025 B_vgg: 0.047 reader cost: 0.63939s batch cost: 2.00111s
```

Notation: In train phase, the `isTrain` value in makeup.yaml file is `True`, but in test phase, its value should be modified as `False`.

### 2.3 Model

Model|Dataset|BatchSize|Inference speed|Download
---|:--:|:--:|:--:|:--:
PSGAN|MT-Dataset| 1 | 1.9s/image (GPU:P40) | [model](https://paddlegan.bj.bcebos.com/models/psgan_weight.pdparams)

## 3. Result
![](../../imgs/makeup_shifter.png)


### 4. References

```
@InProceedings{Jiang_2020_CVPR,
  author = {Jiang, Wentao and Liu, Si and Gao, Chen and Cao, Jie and He, Ran and Feng, Jiashi and Yan, Shuicheng},
  title = {PSGAN: Pose and Expression Robust Spatial-Aware GAN for Customizable Makeup Transfer},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2020}
}
```
