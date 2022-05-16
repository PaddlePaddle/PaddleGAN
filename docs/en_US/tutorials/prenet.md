# PReNet

## 1 Introduction
"Progressive Image Deraining Networks: A Better and Simpler Baseline" provides a better and simpler baseline deraining network by considering network architecture, input and output, and loss functions.

<div align="center">
    <img src="https://github.com/simonsLiang/PReNet_paddle/blob/main/data/net.jpg" width=800">
</div>

## 2 How to use

### 2.1 Prepare dataset

  The dataset(RainH.zip) used by PReNet can be downloaded from [here](https://pan.baidu.com/s/1_vxCatOV3sOA6Vkx1l23eA?pwd=vitu),uncompress it and get two folders(RainTrainH、Rain100H).

  The structure of dataset is as following:

```
    ├── RainH
        ├── RainTrainH
        |    ├── rain
        |    |    ├── 1.png
        |    |    └── 2.png
        |    |        .
        |    |        .
        |    └── norain
        |        ├── 1.png
        |        └── 2.png
        |            .
        |            .
        └── Rain100H
            ├── rain
            |    ├── 001.png
            |    └── 002.png
            |        .
            |        .
            └── norain
                ├── 001.png
                └── 002.png
                    .
                    .
  ```

### 2.2 Train/Test


  train model:
  ```
     python -u tools/main.py --config-file configs/prenet.yaml
  ```

  test model:
  ```
     python tools/main.py --config-file configs/prenet.yaml --evaluate-only --load ${PATH_OF_WEIGHT}
  ```

## 3 Results
Evaluated on RGB channels, scale pixels in each border are cropped before evaluation.

The metrics are PSNR / SSIM.

| Method | Rain100H | 
|---|---|
| PReNet  | 29.5037 / 0.899 | 


Input:

<div align="center">
    <img src="https://github.com/simonsLiang/PReNet_paddle/blob/main/data/rain-001.png" width=300">
</div>

Output:

<div align="center">
    <img src="https://github.com/simonsLiang/PReNet_paddle/blob/main/data/derain-rain-001.png" width=300">
</div>

## 4 Model Download

| model | dataset |
|---|---|
| [PReNet](https://paddlegan.bj.bcebos.com/models/PReNet.pdparams)  | [RainH.zip](https://pan.baidu.com/s/1_vxCatOV3sOA6Vkx1l23eA?pwd=vitu) |





# References

- 1. [Progressive Image Deraining Networks: A Better and Simpler Baseline](https://arxiv.org/pdf/1901.09221v3.pdf)


```
@inproceedings{ren2019progressive,
   title={Progressive Image Deraining Networks: A Better and Simpler Baseline},
   author={Ren, Dongwei and Zuo, Wangmeng and Hu, Qinghua and Zhu, Pengfei and Meng, Deyu},
   booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
   year={2019},
 }

