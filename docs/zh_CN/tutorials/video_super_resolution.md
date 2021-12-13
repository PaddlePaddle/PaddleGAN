
# 视频超分

## 1.1 原理介绍

  视频超分源于图像超分，其目的是从一个或多个低分辨率（LR）图像中恢复高分辨率（HR）图像。它们的区别也很明显，由于视频是由多个帧组成的，所以视频超分通常利用帧间的信息来进行修复。

  这里我们提供百度自研SOTA超分系列模型PP-MSVSR、业界领先视频超分模型[EDVR](https://arxiv.org/pdf/1905.02716.pdf)、[BasicVSR](https://arxiv.org/pdf/2012.02181.pdf)，[IconVSR](https://arxiv.org/pdf/2012.02181.pdf)和[BasicVSR++](https://arxiv.org/pdf/2104.13371v1.pdf)。

### ⭐ PP-MSVSR ⭐
  百度自研的[PP-MSVSR](https://arxiv.org/pdf/2112.02828.pdf)是一种多阶段视频超分深度架构，具有局部融合模块、辅助损失和细化对齐模块，以逐步细化增强结果。具体来说，在第一阶段设计了局部融合模块，在特征传播之前进行局部特征融合, 以加强特征传播中跨帧特征的融合。在第二阶段中引入了一个辅助损失，使传播模块获得的特征保留了更多与HR空间相关的信息。在第三阶段中引入了一个细化的对齐模块，以充分利用前一阶段传播模块的特征信息。大量实验证实，PP-MSVSR在Vid4数据集性能优异，仅使用 1.45M 参数PSNR指标即可达到28.13dB。

  [PP-MSVSR](https://arxiv.org/pdf/2112.02828.pdf)提供两种体积模型，开发者可根据实际场景灵活选择：[PP-MSVSR](https://arxiv.org/pdf/2112.02828.pdf)（参数量1.45M）与[PP-MSVSR-L](https://arxiv.org/pdf/2112.02828.pdf)（参数量7.42）。

### EDVR
  [EDVR](https://arxiv.org/pdf/1905.02716.pdf)模型在NTIRE19视频恢复和增强挑战赛的四个赛道中都赢得了冠军，并以巨大的优势超过了第二名。视频超分的主要难点在于（1）如何在给定大运动的情况下对齐多个帧；（2）如何有效地融合具有不同运动和模糊的不同帧。首先，为了处理大的运动，EDVR模型设计了一个金字塔级联的可变形（PCD）对齐模块，在该模块中，从粗到精的可变形卷积被使用来进行特征级的帧对齐。其次，EDVR使用了时空注意力（TSA）融合模块，该模块在时间和空间上同时应用注意力机制，以强调后续恢复的重要特征。

### BasicVSR
  [BasicVSR](https://arxiv.org/pdf/2012.02181.pdf)在VSR的指导下重新考虑了四个基本模块（即传播、对齐、聚合和上采样）的一些最重要的组件。 通过添加一些小设计，重用一些现有组件，得到了简洁的 BasicVSR。与许多最先进的算法相比，BasicVSR在速度和恢复质量方面实现了有吸引力的改进。 同时，通过添加信息重新填充机制和耦合传播方案以促进信息聚合，BasicVSR 可以扩展为 [IconVSR](https://arxiv.org/pdf/2012.02181.pdf)，IconVSR可以作为未来 VSR 方法的强大基线 .

### BasicVSR++
  [BasicVSR++](https://arxiv.org/pdf/2104.13371v1.pdf)通过提出二阶网格传播和导流可变形对齐来重新设计BasicVSR。通过增强传播和对齐来增强循环框架，BasicVSR++可以更有效地利用未对齐视频帧的时空信息。 在类似的计算约束下，新组件可提高性能。特别是，BasicVSR++ 以相似的参数数量在 PSNR 方面比 BasicVSR 高0.82dB。BasicVSR++ 在NTIRE2021的视频超分辨率和压缩视频增强挑战赛中获得三名冠军和一名亚军。

## 1.2 如何使用

### 1.2.1 数据准备

  这里提供4个视频超分辨率常用数据集，REDS，Vimeo90K，Vid4，UDM10。其中REDS和vimeo90k数据集包括训练集和测试集，Vid4和UDM10为测试数据集。将需要的数据集下载解压后放到``PaddleGAN/data``文件夹下 。

  REDS（[数据下载](https://seungjunnah.github.io/Datasets/reds.html)）数据集是NTIRE19比赛最新提出的高质量（720p）视频数据集，其由240个训练片段、30个验证片段和30个测试片段组成（每个片段有100个连续帧）。由于测试数据集不可用，这里在训练集选择了四个具有代表性的片段（分别为'000', '011', '015', '020'，它们具有不同的场景和动作）作为测试集，用REDS4表示。剩下的训练和验证片段被重新分组为训练数据集（总共266个片段）。

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

  Vimeo90K（[数据下载](http://toflow.csail.mit.edu/)）数据集是Tianfan Xue等人构建的一个用于视频超分、视频降噪、视频去伪影、视频插帧的数据集。Vimeo90K是大规模、高质量的视频数据集，包含从vimeo.com下载的 89,800 个视频剪辑，涵盖了大量场景和动作。

  处理后的数据集 Vimeo90K 的组成形式如下:
  ```
    PaddleGAN
      ├── data
          ├── Vimeo90K
                ├── vimeo_septuplet
                |    |──sequences
                |    └──sep_trainlist.txt
                ├── vimeo_septuplet_BD_matlabLRx4
                |    └──sequences
                └── vimeo_super_resolution_test
                     |──low_resolution
                     |──target
                     └──sep_testlist.txt
              ...
  ```

  Vid4（[数据下载](https://paddlegan.bj.bcebos.com/datasets/Vid4.zip)）数据集是常用的视频超分验证数据集，包含4个视频段。

  处理后的数据集 Vid4 的组成形式如下:
  ```
    PaddleGAN
      ├── data
          ├── Vid4
                ├── BDx4
                └── GT
              ...
  ```

  UDM10（[数据下载](https://paddlegan.bj.bcebos.com/datasets/udm10_paddle.tar)）数据集是常用的视频超分验证数据集，包含10个视频段。

  处理后的数据集 UDM10 的组成形式如下:
  ```
    PaddleGAN
      ├── data
          ├── udm10
                ├── BDx4
                └── GT
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

  训练或测试其他视频超分模型，可以在``PaddleGAN/configs``文件夹下找到对应模型的配置文件，将命令中的配置文件改成该视频超分模型的配置文件即可。


## 1.3 实验结果展示
实验数值结果是在 RGB 通道上进行评估。

度量指标为 PSNR / SSIM.

REDS的测试数据集REDS4上的超分性能对比
| 模型| 参数量（M） | 计算量（G） | REDS4 |
|---|---|---|---|
| EDVR_M_wo_tsa_SRx4  | 3.00 | 223 | 30.4429 / 0.8684 |
| EDVR_M_w_tsa_SRx4  | 3.30 | 232 | 30.5169 / 0.8699 |
| EDVR_L_wo_tsa_SRx4  | 19.42 | 974 | 30.8649 / 0.8761 |
| EDVR_L_w_tsa_SRx4  | 20.63 | 1010 | 30.9336 / 0.8773 |
| BasicVSR_x4  | 6.29 | 374 | 31.4325 / 0.8913 |
| IconVSR_x4  | 8.69 | 516 | 31.6882 / 0.8950 |
| BasicVSR++_x4  | 7.32 | 406 | 32.4018 / 0.9071 |
| PP-MSVSR_reds_x4  | 1.45 | 111 | 31.2535 / 0.8884 |
| PP-MSVSR-L_reds_x4  | 7.42 | 543 | 32.5321 / 0.9083 |

REDS的测试数据集REDS4上的去模糊性能对比
| 模型 | REDS4 |
|---|---|
| EDVR_L_wo_tsa_deblur  | 34.9587 / 0.9509 |
| EDVR_L_w_tsa_deblur  | 35.1473 / 0.9526 |

Vimeo90K，Vid4，UDM10测试数据集上超分性能对比
| 模型 | Vimeo90K | Vid4 | UDM10 |
|---|---|---|---|
| PP-MSVSR_vimeo90k_x4 |37.54/0.9499|28.13/0.8604|40.06/0.9699|

## 1.4 模型下载
| 模型 | 数据集 | 下载地址 |
|---|---|---|
| EDVR_M_wo_tsa_SRx4  | REDS | [EDVR_M_wo_tsa_SRx4](https://paddlegan.bj.bcebos.com/models/EDVR_M_wo_tsa_SRx4.pdparams)
| EDVR_M_w_tsa_SRx4  | REDS | [EDVR_M_w_tsa_SRx4](https://paddlegan.bj.bcebos.com/models/EDVR_M_w_tsa_SRx4.pdparams)
| EDVR_L_wo_tsa_SRx4  | REDS | [EDVR_L_wo_tsa_SRx4](https://paddlegan.bj.bcebos.com/models/EDVR_L_wo_tsa_SRx4.pdparams)
| EDVR_L_w_tsa_SRx4  | REDS | [EDVR_L_w_tsa_SRx4](https://paddlegan.bj.bcebos.com/models/EDVR_L_w_tsa_SRx4.pdparams)
| EDVR_L_wo_tsa_deblur  | REDS | [EDVR_L_wo_tsa_deblur](https://paddlegan.bj.bcebos.com/models/EDVR_L_wo_tsa_deblur.pdparams)
| EDVR_L_w_tsa_deblur  | REDS | [EDVR_L_w_tsa_deblur](https://paddlegan.bj.bcebos.com/models/EDVR_L_w_tsa_deblur.pdparams)
| BasicVSR_x4  | REDS | [BasicVSR_x4](https://paddlegan.bj.bcebos.com/models/BasicVSR_reds_x4.pdparams)
| IconVSR_x4  | REDS | [IconVSR_x4](https://paddlegan.bj.bcebos.com/models/IconVSR_reds_x4.pdparams)
| BasicVSR++_x4  | REDS | [BasicVSR++_x4](https://paddlegan.bj.bcebos.com/models/BasicVSR%2B%2B_reds_x4.pdparams)
| PP-MSVSR_reds_x4  | REDS | [PP-MSVSR_reds_x4](https://paddlegan.bj.bcebos.com/models/PP-MSVSR_reds_x4.pdparams)
| PP-MSVSR-L_reds_x4  | REDS | [PP-MSVSR-L_reds_x4](https://paddlegan.bj.bcebos.com/models/PP-MSVSR-L_reds_x4.pdparams)
| PP-MSVSR_vimeo90k_x4  | Vimeo90K | [PP-MSVSR_vimeo90k_x4](https://paddlegan.bj.bcebos.com/models/PP-MSVSR_vimeo90k_x4.pdparams)

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
- 2. [BasicVSR: The Search for Essential Components in Video Super-Resolution and Beyond](https://arxiv.org/pdf/2012.02181.pdf)

  ```
  @InProceedings{chan2021basicvsr,
    author = {Chan, Kelvin C.K. and Wang, Xintao and Yu, Ke and Dong, Chao and Loy, Chen Change},
    title = {BasicVSR: The Search for Essential Components in Video Super-Resolution and Beyond},
    booktitle = {Proceedings of the IEEE conference on computer vision and pattern recognition},
    year = {2021}
    }
  ```
- 3. [BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment](https://arxiv.org/pdf/2104.13371v1.pdf)

  ```
  @article{chan2021basicvsr++,
    author = {Chan, Kelvin C.K. and Zhou, Shangchen and Xu, Xiangyu and Loy, Chen Change},
    title = {BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment},
    booktitle = {arXiv preprint arXiv:2104.13371},
    year = {2021}
    }
  ```

- 4. [PP-MSVSR: Multi-Stage Video Super-Resolution](https://arxiv.org/pdf/2112.02828.pdf)

  ```
  @article{jiang2021PP-MSVSR,
    author = {Jiang, Lielin and Wang, Na and Dang, Qingqing and Liu, Rui and Lai, Baohua},
    title = {PP-MSVSR: Multi-Stage Video Super-Resolution},
    booktitle = {arXiv preprint arXiv:2112.02828},
    year = {2021}
    }
  ```
