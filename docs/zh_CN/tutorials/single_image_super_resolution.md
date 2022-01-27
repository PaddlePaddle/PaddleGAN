# 1 单张图像超分

## 1.1 原理介绍

  超分是放大和改善图像细节的过程。它通常将低分辨率图像作为输入，将同一图像放大到更高分辨率作为输出。这里我们提供了四种超分辨率模型，即[RealSR](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w31/Ji_Real-World_Super-Resolution_via_Kernel_Estimation_and_Noise_Injection_CVPRW_2020_paper.pdf), [ESRGAN](https://arxiv.org/abs/1809.00219v2), [LESRCNN](https://arxiv.org/abs/2007.04344),[PAN](https://arxiv.org/pdf/2010.01073.pdf).
  [RealSR](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w31/Ji_Real-World_Super-Resolution_via_Kernel_Estimation_and_Noise_Injection_CVPRW_2020_paper.pdf)通过估计各种模糊内核以及实际噪声分布，为现实世界的图像设计一种新颖的真实图片降采样框架。基于该降采样框架，可以获取与真实世界图像共享同一域的低分辨率图像。RealSR是一个旨在提高感知度的真实世界超分辨率模型。对合成噪声数据和真实世界图像进行的大量实验表明，RealSR模型能够有效降低了噪声并提高了视觉质量。
  [ESRGAN](https://arxiv.org/abs/1809.00219v2)是增强型SRGAN，为了进一步提高SRGAN的视觉质量，ESRGAN在SRGAN的基础上改进了SRGAN的三个关键组件。此外，ESRGAN还引入了未经批量归一化的剩余密集块（RRDB）作为基本的网络构建单元，让鉴别器预测相对真实性而不是绝对值，并利用激活前的特征改善感知损失。得益于这些改进，提出的ESRGAN实现了比SRGAN更好的视觉质量和更逼真、更自然的纹理，并在PIRM2018-SR挑战赛中获得第一名。
  考虑到CNNs在SISR的应用上往往会消耗大量的计算量和存储空间来训练SR模型。轻量级增强SR-CNN（[LESRCNN](https://arxiv.org/abs/2007.04344)）被提出。大量实验表明，LESRCNN在定性和定量评价方面优于现有的SISR算法。
  之后[PAN](https://arxiv.org/pdf/2010.01073.pdf)设计了一种用于图像超分辨率（SR）的轻量级卷积神经网络。



## 1.2 如何使用

### 1.2.1 数据准备

  常用的图像超分数据集如下：
  | name | 数据集 | 数据描述 | 下载 |
  |---|---|---|---|
  | 2K Resolution  | [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) | proposed in [NTIRE17](https://data.vision.ee.ethz.ch/cvl/ntire17//) (800 train and 100 validation) | [official website](https://data.vision.ee.ethz.ch/cvl/DIV2K/) |
  | Classical SR Testing  | Set5 | Set5 test dataset | [Google Drive](https://drive.google.com/drive/folders/1B3DJGQKB6eNdwuQIhdskA64qUuVKLZ9u) / [Baidu Drive](https://pan.baidu.com/s/1q_1ERCMqALH0xFwjLM0pTg#list/path=%2Fsharelink2016187762-785433459861126%2Fclassical_SR_datasets&parentPath=%2Fsharelink2016187762-785433459861126) |
  | Classical SR Testing  | Set14 | Set14 test dataset | [Google Drive](https://drive.google.com/drive/folders/1B3DJGQKB6eNdwuQIhdskA64qUuVKLZ9u) / [Baidu Drive](https://pan.baidu.com/s/1q_1ERCMqALH0xFwjLM0pTg#list/path=%2Fsharelink2016187762-785433459861126%2Fclassical_SR_datasets&parentPath=%2Fsharelink2016187762-785433459861126) |

  数据集DIV2K, Set5 和 Set14 的组成形式如下:
  ```
    PaddleGAN
      ├── data
          ├── DIV2K
                ├── DIV2K_train_HR
                ├── DIV2K_train_LR_bicubic
                |    ├──X2
                |    ├──X3
                |    └──X4
                ├── DIV2K_valid_HR
                ├── DIV2K_valid_LR_bicubic
              Set5
                ├── GTmod12
                ├── LRbicx2
                ├── LRbicx3
                ├── LRbicx4
                └── original
              Set14
                ├── GTmod12
                ├── LRbicx2
                ├── LRbicx3
                ├── LRbicx4
                └── original
              ...
  ```
  使用以下命令处理DIV2K数据集:
  ```
    python data/process_div2k_data.py --data-root data/DIV2K
  ```
  程序完成后，检查DIV2K目录中是否有``DIV2K_train_HR_sub``、``X2_sub``、``X3_sub``和``X4_sub``目录
  ```
    PaddleGAN
      ├── data
          ├── DIV2K
                ├── DIV2K_train_HR
                ├── DIV2K_train_HR_sub
                ├── DIV2K_train_LR_bicubic
                |    ├──X2
                |    ├──X2_sub
                |    ├──X3
                |    ├──X3_sub
                |    ├──sX4
                |    └──X4_sub
                ├── DIV2K_valid_HR
                ├── DIV2K_valid_LR_bicubic
              ...
  ```

#### Realsr df2k model的数据准备

  从 [NTIRE 2020 RWSR](https://competitions.codalab.org/competitions/22220#participate) 下载数据集并解压到您的路径下。
  将 Corrupted-tr-x.zip 和 Corrupted-tr-y.zip 解压到 ``PaddleGAN/data/ntire20`` 目录下。

  运行如下命令:
  ```
    python ./data/realsr_preprocess/create_bicubic_dataset.py --dataset df2k --artifacts tdsr
    python ./data/realsr_preprocess/collect_noise.py --dataset df2k --artifacts tdsr
  ```

### 1.2.2 训练/测试

  示例以df2k数据集和RealSR模型为例。如果您想使用自己的数据集，可以在配置文件中修改数据集为您自己的数据集。如果您想使用其他模型，可以通过替换配置文件。

  训练模型:
  ```
     python -u tools/main.py --config-file configs/realsr_bicubic_noise_x4_df2k.yaml
  ```

  测试模型:
  ```
     python tools/main.py --config-file configs/realsr_bicubic_noise_x4_df2k.yaml --evaluate-only --load ${PATH_OF_WEIGHT}
  ```

## 1.3 实验结果展示
实验数值结果是在 RGB 通道上进行评估，并在评估之前裁剪每个边界的尺度像素。

度量指标为 PSNR / SSIM.

| 模型 | Set5 | Set14 | DIV2K |
|---|---|---|---|
| realsr_df2k  | 28.4385 / 0.8106 | 24.7424 / 0.6678 | 26.7306 / 0.7512 |
| realsr_dped  | 20.2421 / 0.6158 | 19.3775 / 0.5259 | 20.5976 / 0.6051 |
| realsr_merge  | 24.8315 / 0.7030 | 23.0393 / 0.5986 | 24.8510 / 0.6856 |
| lesrcnn_x4  | 31.9476 / 0.8909 | 28.4110 / 0.7770 | 30.231 / 0.8326 |
| esrgan_psnr_x4  | 32.5512 / 0.8991 | 28.8114 / 0.7871 | 30.7565 / 0.8449 |
| esrgan_x4  | 28.7647 / 0.8187 | 25.0065 / 0.6762 | 26.9013 / 0.7542 |
| pan_x4  | 30.4574 / 0.8643 | 26.7204 / 0.7434 | 28.9187 / 0.8176 |
| drns_x4  | 32.6684 / 0.8999 | 28.9037 / 0.7885 | - |

PAN指标对比

paddle模型使用DIV2K数据集训练，torch模型使用df2k和DIV2K数据集训练。

| 框架 | Set5 | Set14 |
|---|---|---|
| paddle  | 30.4574 / 0.8643 | 26.7204 / 0.7434 |
| torch  | 30.2183 / 0.8643 | 26.8035 / 0.7445 |

<!-- ![](../../imgs/horse2zebra.png) -->


## 1.4 模型下载
| 模型 | 数据集 | 下载地址 |
|---|---|---|
| realsr_df2k  | df2k | [realsr_df2k](https://paddlegan.bj.bcebos.com/models/realsr_df2k.pdparams)
| realsr_dped  | dped | [realsr_dped](https://paddlegan.bj.bcebos.com/models/realsr_dped.pdparams)
| realsr_merge  | DIV2K | [realsr_merge](https://paddlegan.bj.bcebos.com/models/realsr_merge.pdparams)
| lesrcnn_x4  | DIV2K | [lesrcnn_x4](https://paddlegan.bj.bcebos.com/models/lesrcnn_x4.pdparams)
| esrgan_psnr_x4  | DIV2K | [esrgan_psnr_x4](https://paddlegan.bj.bcebos.com/models/esrgan_psnr_x4.pdparams)
| esrgan_x4  | DIV2K | [esrgan_x4](https://paddlegan.bj.bcebos.com/models/esrgan_x4.pdparams)
| pan_x4  | DIV2K | [pan_x4](https://paddlegan.bj.bcebos.com/models/pan_x4.pdparams)
| drns_x4  | DIV2K | [drns_x4](https://paddlegan.bj.bcebos.com/models/DRNSx4.pdparams)


# 参考文献

- 1. [Real-World Super-Resolution via Kernel Estimation and Noise Injection](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w31/Ji_Real-World_Super-Resolution_via_Kernel_Estimation_and_Noise_Injection_CVPRW_2020_paper.pdf)

  ```
  @inproceedings{ji2020real,
  title={Real-World Super-Resolution via Kernel Estimation and Noise Injection},
  author={Ji, Xiaozhong and Cao, Yun and Tai, Ying and Wang, Chengjie and Li, Jilin and Huang, Feiyue},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  pages={466--467},
  year={2020}
  }
  ```

- 2. [ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks](https://arxiv.org/abs/1809.00219v2)

  ```
  @inproceedings{wang2018esrgan,
  title={Esrgan: Enhanced super-resolution generative adversarial networks},
  author={Wang, Xintao and Yu, Ke and Wu, Shixiang and Gu, Jinjin and Liu, Yihao and Dong, Chao and Qiao, Yu and Change Loy, Chen},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={0--0},
  year={2018}
  }
  ```

- 3. [Lightweight image super-resolution with enhanced CNN](https://arxiv.org/abs/2007.04344)

  ```
  @article{tian2020lightweight,
  title={Lightweight image super-resolution with enhanced CNN},
  author={Tian, Chunwei and Zhuge, Ruibin and Wu, Zhihao and Xu, Yong and Zuo, Wangmeng and Chen, Chen and Lin, Chia-Wen},
  journal={Knowledge-Based Systems},
  volume={205},
  pages={106235},
  year={2020},
  publisher={Elsevier}
  }
  ```
- 4. [Efficient Image Super-Resolution Using Pixel Attention](https://arxiv.org/pdf/2010.01073.pdf)

  ```
  @inproceedings{Hengyuan2020Efficient,
  title={Efficient Image Super-Resolution Using Pixel Attention},
  author={Hengyuan Zhao and Xiangtao Kong and Jingwen He and Yu Qiao and Chao Dong},
  booktitle={Computer Vision – ECCV 2020 Workshops},
  volume={12537},
  pages={56-72},
  year={2020}
  }
  ```
  - 5. [Closed-loop Matters: Dual Regression Networks for Single Image Super-Resolution](https://arxiv.org/pdf/2003.07018.pdf)

  ```
  @inproceedings{guo2020closed,
  title={Closed-loop Matters: Dual Regression Networks for Single Image Super-Resolution},
  author={Guo, Yong and Chen, Jian and Wang, Jingdong and Chen, Qi and Cao, Jiezhang and Deng, Zeshuai and Xu, Yanwu and Tan, Mingkui},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
  }
  ```
