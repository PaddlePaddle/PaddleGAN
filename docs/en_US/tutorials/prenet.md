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
    ├── data
        ├── RainTrainH
        └── Rain100H
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

![](https://user-images.githubusercontent.com/79366697/146308440-65259d70-d056-43d4-8cf5-a82530993910.jpg)

## 4 Model Download
| 模型 | 数据集 | 下载地址 |
|---|---|---|
| PReNet(net_latest.pdparams)  | RainH.zip | [RainH.zip](https://pan.baidu.com/s/1_vxCatOV3sOA6Vkx1l23eA?pwd=vitu)




# References

- 1. [Progressive Image Deraining Networks: A Better and Simpler Baseline](https://arxiv.org/pdf/1901.09221v3.pdf)

  ```
  ```
