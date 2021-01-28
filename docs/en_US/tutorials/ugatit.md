# 1 U-GAT-IT

## 1.1 Principle

  Similar to CycleGAN, [U-GAT-IT](https://arxiv.org/abs/1907.10830) uses unpaired pictures for image translation, input two different images with different styles, and automatically perform style transfer. Differently, U-GAT-IT is a novel method for unsupervised image-to-image translation, which incorporates a new attention module and a new learnable normalization function in an end-to-end manner. 

## 1.2 How to use  

### 1.2.1 Prepare Datasets

  Selfie2anime dataset used by U-GAT-IT can be download from [here](https://www.kaggle.com/arnaud58/selfie2anime). You can also use your own dataset.
  The structure of dataset is as following:
  ```
    ├── dataset
        └── YOUR_DATASET_NAME
            ├── trainA
            ├── trainB
            ├── testA
            └── testB
  ```

### 1.2.2 Train/Test

  Datasets used in example is selfie2anime, you can change it to your own dataset in the config file.

  Train a model:
  ```
     python -u tools/main.py --config-file configs/ugatit_selfie2anime_light.yaml
  ```

  Test the model:
  ```
     python tools/main.py --config-file configs/ugatit_selfie2anime_light.yaml --evaluate-only --load ${PATH_OF_WEIGHT}
  ```

## 1.3 Results

![](../../imgs/ugatit.png)

## 1.4 模型下载
| 模型 | 数据集 | 下载地址 |
|---|---|---|
| ugatit_light  | selfie2anime | [ugatit_light](https://paddlegan.bj.bcebos.com/models/ugatit_light.pdparams)




# References

- 1. [U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation](https://arxiv.org/abs/1907.10830)

  ```
  @article{kim2019u,
  title={U-GAT-IT: unsupervised generative attentional networks with adaptive layer-instance normalization for image-to-image translation},
  author={Kim, Junho and Kim, Minjae and Kang, Hyeonwoo and Lee, Kwanghee},
  journal={arXiv preprint arXiv:1907.10830},
  year={2019}
  }
  ```

