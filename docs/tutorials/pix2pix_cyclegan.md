# Pix2pix

[Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004)

## Prepare Datasets

  Paired datasets used by Pix2pix can be download from [here](http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/)
  For example, the structure of facades is as following:
    facades
       ├── test
       ├── train
       └── val

## Train/Test

  Datasets used in example is facades, you can change it to your own dataset in the config file.

  Train a model:
  ```
     python -u tools/main.py --config-file configs/pix2pix_facades.yaml
  ```

  Test the model:
  ```
     python tools/main.py --config-file configs/pix2pix_facades.yaml --evaluate-only --load ${PATH_OF_WEIGHT}
  ```

## Results

![](../imgs/horse2zebra.png)

## Principle

  Pix2pix is a general-purpose solution to image-to-image translation problems, these networks not only learn the mapping from input image to output image, but also learn a loss function to train this mapping. 

    


# CycleGAN

[Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)

## Prepare Datasets

  Unpair datasets used by CycleGAN can be download from [here](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/)
  For example, the structure of cityscapes is as following:
    cityscapes
        ├── test
        ├── testA
        ├── testB
        ├── train
        ├── trainA
        └── trainB

## Train/Test

  Datasets used in example is cityscapes, you can change it to your own dataset in the config file.

  Train a model:
  ```
     python -u tools/main.py --config-file configs/cyclegan_cityscapes.yaml
  ```

  Test the model:
  ```
     python tools/main.py --config-file configs/cyclegan_cityscapes.yaml --evaluate-only --load ${PATH_OF_WEIGHT}
  ```

## Results

![](../imgs/A2B.png)

## Principle

  CycleGAN learns a mapping G : X → Y such that the distribution of images from G(X) is indistinguishable from the distribution Y using an adversarial loss.
