# Face Enhancement

## 1. face enhancement introduction

Blind face restoration (BFR) from severely degraded face images in the wild is a very challenging problem. Due to the high illness of the problem and the complex unknown degradation, directly training a deep neural network (DNN) usually cannot lead to acceptable results. Existing generative adversarial network (GAN) based methods can produce better results but tend to generate over-smoothed restorations. Here we provide the [GPEN](https://arxiv.org/abs/2105.06070) model. GPEN was proposed by first learning a GAN for high-quality face image generation and embedding it into a U-shaped DNN as a prior decoder, then fine-tuning the GAN prior embedded DNN with a set of synthesized low-quality face images. The GAN blocks are designed to ensure that the latent code and noise input to the GAN can be respectively generated from the deep and shallow features of the DNN, controlling the global face structure, local face details and background of the reconstructed image. The proposed GAN prior embedded network (GPEN) is easy-to-implement, and it can generate visually photo-realistic results. Experiments demonstrated that the proposed GPEN achieves significantly superior results to state-of-the-art BFR methods both quantitatively and qualitatively, especially for the restoration of severely degraded face images in the wild.

## How to use

### face enhancement

The user could use the following command to do face enhancement and select the local image as input：

```python
import paddle
from ppgan.faceutils.face_enhancement import FaceEnhancement

faceenhancer = FaceEnhancement()
img = faceenhancer.enhance_from_image(img)
```

Note: please convert the image to float type, currently does not support int8 type.

### Train (TODO)

In the future, training scripts will be added to facilitate users to train more types of GPEN.

## Results

![1](https://user-images.githubusercontent.com/79366697/146891109-d204497f-7e71-4899-bc65-e1b101ce6293.jpg)

## Reference

```
@inproceedings{inproceedings,
author = {Yang, Tao and Ren, Peiran and Xie, Xuansong and Zhang, Lei},
year = {2021},
month = {06},
pages = {672-681},
title = {GAN Prior Embedded Network for Blind Face Restoration in the Wild},
doi = {10.1109/CVPR46437.2021.00073}
}

```
