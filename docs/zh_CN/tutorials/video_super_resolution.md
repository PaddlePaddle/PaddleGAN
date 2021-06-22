
# 1 视频超分

## 1.1 原理介绍

  视频超分源于图像超分，其目的是从一个或多个低分辨率（LR）图像中恢复高分辨率（HR）图像。它们的区别也很明显，由于视频是由多个帧组成的，所以视频超分通常利用帧间的信息来进行修复。这里我们提供视频超分模型[EDVR](https://arxiv.org/pdf/1905.02716.pdf).

  [EDVR](https://arxiv.org/pdf/1905.02716.pdf)模型在NTIRE19视频恢复和增强挑战赛的四个赛道中都赢得了冠军，并以巨大的优势超过了第二名。视频超分的主要难点在于（1）如何在给定大运动的情况下对齐多个帧；（2）如何有效地融合具有不同运动和模糊的不同帧。首先，为了处理大的运动，EDVR模型设计了一个金字塔级联的可变形（PCD）对齐模块，在该模块中，从粗到精的可变形卷积被使用来进行特征级的帧对齐。其次，EDVR使用了时空注意力（TSA）融合模块，该模块在时间和空间上同时应用注意力机制，以强调后续恢复的重要特征。



## 1.2 如何使用

### 1.2.1 数据准备

  REDS（[数据下载](https://seungjunnah.github.io/Datasets/reds.html)）数据集是NTIRE19公司最新提出的高质量（720p）视频数据集，其由240个训练片段、30个验证片段和30个测试片段组成（每个片段有100个连续帧）。由于测试数据集不可用，这里在训练集选择了四个具有代表性的片段（分别为'000', '011', '015', '020'，它们具有不同的场景和动作）作为测试集，用REDS4表示。剩下的训练和验证片段被重新分组为训练数据集（总共266个片段）。

  处理后的数据集 REDS 的组成形式如下:
  ```
    PaddleGAN
      ├── data
          ├── REDS
                ├── train_sharp
                |    └──X4
                ├── train_sharp_bicubic
                |    └──X4
                ├── REDS4_test_sharp
                |    └──X4
                └── REDS4_test_sharp_bicubic
                     └──X4
              ...
  ```

### 1.2.2 训练/测试

  EDVR模型根据模型中间通道数分为EDVR_L(128通道)和EDVR_M(64通道)两种模型。下面以EDVR_M模型为例介绍模型训练与测试。

  EDVR模型训练一般分两个阶段训练，先不带TSA模块训练，训练与测试命令如下:

  训练模型:
  ```
     python -u tools/main.py --config-file configs/edvr_m_wo_tsa.yaml
  ```

  测试模型:
  ```
     python tools/main.py --config-file configs/edvr_m_wo_tsa.yaml --evaluate-only --load ${PATH_OF_WEIGHT_WITHOUT_TSA}
  ```

  然后用保存的不带TSA模块的EDVR权重作为EDVR模型的初始化，训练完整的EDVR模型，训练与测试命令如下:

  训练模型:
  ```
     python -u tools/main.py --config-file configs/edvr_m_w_tsa.yaml --load ${PATH_OF_WEIGHT_WITHOUT_TSA}
  ```

  测试模型:
  ```
     python tools/main.py --config-file configs/edvr_m_w_tsa.yaml --evaluate-only --load ${PATH_OF_WEIGHT}
  ```


## 1.3 实验结果展示
实验数值结果是在 RGB 通道上进行评估。

度量指标为 PSNR / SSIM.

| 模型 | REDS4 |
|---|---|
| EDVR_M_wo_tsa_SRx4  | 30.4429 / 0.8684 |
| EDVR_M_w_tsa_SRx4  | 30.5169 / 0.8699 |
| EDVR_L_wo_tsa_SRx4  | 30.8649 / 0.8761 |
| EDVR_L_w_tsa_SRx4  | 30.9336 / 0.8773 |


## 1.4 模型下载
| 模型 | 数据集 | 下载地址 |
|---|---|---|
| EDVR_M_wo_tsa_SRx4  | REDS | [EDVR_M_wo_tsa_SRx4](https://paddlegan.bj.bcebos.com/models/EDVR_M_wo_tsa_SRx4.pdparams)
| EDVR_M_w_tsa_SRx4  | REDS | [EDVR_M_w_tsa_SRx4](https://paddlegan.bj.bcebos.com/models/EDVR_M_w_tsa_SRx4.pdparams)
| EDVR_L_wo_tsa_SRx4  | REDS | [EDVR_L_wo_tsa_SRx4](https://paddlegan.bj.bcebos.com/models/EDVR_L_wo_tsa_SRx4.pdparams)
| EDVR_L_w_tsa_SRx4  | REDS | [EDVR_L_w_tsa_SRx4](https://paddlegan.bj.bcebos.com/models/EDVR_L_w_tsa_SRx4.pdparams)





# 参考文献

- 1. [EDVR: Video Restoration with Enhanced Deformable Convolutional Networks](https://arxiv.org/pdf/1905.02716.pdf)

  ```
  @InProceedings{wang2019edvr,
    author = {Wang, Xintao and Chan, Kelvin C.K. and Yu, Ke and Dong, Chao and Loy, Chen Change},
    title = {EDVR: Video Restoration with Enhanced Deformable Convolutional Networks},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month = {June},
    year = {2019}
    }
  ```
