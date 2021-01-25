# 1 Super Resolution

## 1.1 Principle

  Super resolution is a process of upscaling and improving the details within an image. It usually takes a low-resolution image as input and upscales the same image to a higher resolution as output.
  Here we provide three super-resolution models, namely [RealSR](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w31/Ji_Real-World_Super-Resolution_via_Kernel_Estimation_and_Noise_Injection_CVPRW_2020_paper.pdf), [ESRGAN](https://arxiv.org/abs/1809.00219v2), [LESRCNN](https://arxiv.org/abs/2007.04344).
  [RealSR](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w31/Ji_Real-World_Super-Resolution_via_Kernel_Estimation_and_Noise_Injection_CVPRW_2020_paper.pdf) proposed a realworld super-resolution model aiming at better perception.
  [ESRGAN](https://arxiv.org/abs/1809.00219v2) is an enhanced SRGAN that improves the three key components of SRGAN.
  [LESRCNN](https://arxiv.org/abs/2007.04344) is a lightweight enhanced SR CNN (LESRCNN) with three successive sub-blocks.


## 1.2 How to use  

### 1.2.1 Prepare Datasets

  A list of common image super-resolution datasets is as following:
  | Name | Datasets | Short Description | Download |
  |---|---|---|---|
  | 2K Resolution  | [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) | proposed in [NTIRE17](https://data.vision.ee.ethz.ch/cvl/ntire17//) (800 train and 100 validation) | [official website](https://data.vision.ee.ethz.ch/cvl/DIV2K/) |
  | Classical SR Testing  | Set5 | Set5 test dataset | [Google Drive](https://drive.google.com/drive/folders/1B3DJGQKB6eNdwuQIhdskA64qUuVKLZ9u) / [Baidu Drive](https://pan.baidu.com/s/1q_1ERCMqALH0xFwjLM0pTg#list/path=%2Fsharelink2016187762-785433459861126%2Fclassical_SR_datasets&parentPath=%2Fsharelink2016187762-785433459861126) |
  | Classical SR Testing  | Set14 | Set14 test dataset | [Google Drive](https://drive.google.com/drive/folders/1B3DJGQKB6eNdwuQIhdskA64qUuVKLZ9u) / [Baidu Drive](https://pan.baidu.com/s/1q_1ERCMqALH0xFwjLM0pTg#list/path=%2Fsharelink2016187762-785433459861126%2Fclassical_SR_datasets&parentPath=%2Fsharelink2016187762-785433459861126) |

  The structure of DIV2K is as following:
  ```
    DIV2K
       ├── DIV2K_train_HR
       ├── DIV2K_train_LR_bicubic
       |    ├──X2
       |    ├──X3
       |    └──X4
       ├── DIV2K_valid_HR
       ├── DIV2K_valid_LR_bicubic
       ...
  ```

  The structures of Set5 and Set14 are similar. Taking Set5 as an example, the structure is as following:
  ```
    Set5
      ├── GTmod12
      ├── LRbicx2
      ├── LRbicx3
      ├── LRbicx4
      └── original
  ```


### 1.2.2 Train/Test

  Datasets used in example is df2k, you can change it to your own dataset in the config file. The model used in example is RealSR, you can change other models by replacing the config file.

  Train a model:
  ```
     python -u tools/main.py --config-file configs/realsr_bicubic_noise_x4_df2k.yaml
  ```

  Test the model:
  ```
     python tools/main.py --config-file configs/realsr_bicubic_noise_x4_df2k.yaml --evaluate-only --load ${PATH_OF_WEIGHT}
  ```

## 1.3 Results
Evaluated on RGB channels, scale pixels in each border are cropped before evaluation.

The metrics are PSNR / SSIM.

| Method | Set5 | Set14 | DIV2K |
|---|---|---|---|
| realsr_df2k  | 28.4385 / 0.8106 | 24.7424 / 0.6678 | 26.7306 / 0.7512 |
| realsr_dped  | 20.2421 / 0.6158 | 19.3775 / 0.5259 | 20.5976 / 0.6051 |
| realsr_merge  | 24.8315 / 0.7030 | 23.0393 / 0.5986 | 24.8510 / 0.6856 |
| lesrcnn_x4  | 31.9476 / 0.8909 | 28.4110 / 0.7770 | 30.231 / 0.8326 |
| esrgan_psnr_x4  | 32.5512 / 0.8991 | 28.8114 / 0.7871 | 30.7565 / 0.8449 |
| esrgan_x4  | 28.7647 / 0.8187 | 25.0065 / 0.6762 | 26.9013 / 0.7542 |


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
| drns_x4  | DIV2K | [drns_x4](https://paddlegan.bj.bcebos.com/models/DRNSx4.pdparams)


# References

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
- 4. [Closed-loop Matters: Dual Regression Networks for Single Image Super-Resolution](https://arxiv.org/pdf/2003.07018.pdf)

  ```
  @inproceedings{guo2020closed,
  title={Closed-loop Matters: Dual Regression Networks for Single Image Super-Resolution},
  author={Guo, Yong and Chen, Jian and Wang, Jingdong and Chen, Qi and Cao, Jiezhang and Deng, Zeshuai and Xu, Yanwu and Tan, Mingkui},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
  ```
