
# 1 Video Super Resolution (VSR)

## 1.1 Principle

  Video super-resolution originates from image super-resolution, which aims to recover high-resolution (HR) images from one or more low resolution (LR) images. The difference between them is that the video is composed of multiple frames, so the video super-resolution usually uses the information between frames to repair. Here we provide the video super-resolution model [EDVR](https://arxiv.org/pdf/1905.02716.pdf), [BasicVSR](https://arxiv.org/pdf/2012.02181.pdf),[IconVSR](https://arxiv.org/pdf/2012.02181.pdf),[BasicVSR++](https://arxiv.org/pdf/2104.13371v1.pdf), and PP-MSVSR.

### 🔥 PP-MSVSR 🔥
  [PP-MSVSR](https://arxiv.org/pdf/2112.02828.pdf) is a multi-stage VSR deep architecture,  with local fusion module, auxiliary loss and refined align module to refine the enhanced result progressively. Specifically, in order to strengthen the fusion of features across frames in feature propagation, a local fusion module is designed in stage-1 to perform local feature fusion before feature propagation. Moreover, an auxiliary loss in stage-2 is introduced to make the features obtained by the propagation module reserve more correlated information connected to the HR space, and introduced a refined align module in stage-3 to make full use of the feature information of the previous stage. Extensive experiments substantiate that PP-MSVSR achieves a promising performance of Vid4 datasets, which PSNR metric can achieve 28.13 with only 1.45M parameters.

  Additionally, [PP-MSVSR](https://arxiv.org/pdf/2112.02828.pdf) provides two different models with 1.45M and 7.4M parameters in order to satisfy different requirements.

### EDVR
  [EDVR](https://arxiv.org/pdf/1905.02716.pdf) wins the champions and outperforms the second place by a large margin in all four tracks in the NTIRE19 video restoration and enhancement challenges. The main difficulties of video super-resolution from two aspects: (1) how to align multiple frames given large motions, and (2) how to effectively fuse different frames with diverse motion and blur. First, to handle large motions, EDVR devise a Pyramid, Cascading and Deformable (PCD) alignment module, in which frame alignment is done at the feature level using deformable convolutions in a coarse-to-fine manner. Second, EDVR propose a Temporal and Spatial Attention (TSA) fusion module, in which attention is applied both temporally and spatially, so as to emphasize important features for subsequent restoration.

  [BasicVSR](https://arxiv.org/pdf/2012.02181.pdf) reconsiders some most essential components for VSR guided by four basic functionalities, i.e., Propagation, Alignment, Aggregation, and Upsampling. By reusing some existing components added with minimal redesigns, a succinct pipeline, BasicVSR, achieves appealing improvements in terms of speed and restoration quality in comparison to many state-of-the-art algorithms. By presenting an informationrefill mechanism and a coupled propagation scheme to facilitate information aggregation, the BasicVSR can be expanded to [IconVSR](https://arxiv.org/pdf/2012.02181.pdf), which can serve as strong baselines for future VSR approaches.

  [BasicVSR++](https://arxiv.org/pdf/2104.13371v1.pdf) redesign BasicVSR by proposing second-order grid propagation and flowguided deformable alignment. By empowering the recurrent framework with the enhanced propagation and alignment, BasicVSR++ can exploit spatiotemporal information across misaligned video frames more effectively. The new components lead to an improved performance under a similar computational constraint. In particular, BasicVSR++ surpasses BasicVSR by 0.82 dB in PSNR with similar number of parameters. In NTIRE 2021, BasicVSR++ obtains three champions and one runner-up in the Video Super-Resolution and Compressed Video Enhancement Challenges.



## 1.2 How to use  

### 1.2.1 Prepare Datasets
  Here are 4 commonly used video super-resolution dataset, REDS, Vimeo90K, Vid4, UDM10. The REDS and Vimeo90K dataset include train dataset and test dataset, Vid4 and UDM10 are test dataset. Download and decompress the required dataset and place it under the ``PaddleGAN/data``.

  REDS（[download](https://seungjunnah.github.io/Datasets/reds.html)）is a newly proposed high-quality (720p) video dataset in the NTIRE19 Competition. REDS consists of 240 training clips, 30 validation clips and 30 testing clips (each with 100 consecutive frames). Since the test ground truth is not available, we select four representative clips (they are '000', '011', '015', '020', with diverse scenes and motions) as our test set, denoted by REDS4. The remaining training and validation clips are re-grouped as our training dataset (a total of 266 clips).

  The structure of the processed REDS is as follows:
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

  Vimeo90K ([download](http://toflow.csail.mit.edu/)) is designed by Tianfan Xue etc. for the following four video processing tasks: temporal frame interpolation, video denoising, video deblocking, and video super-resolution. Vimeo90K is a large-scale, high-quality video dataset. This dataset consists of 89,800 video clips downloaded from vimeo.com, which covers large variaty of scenes and actions.

  The structure of the processed Vimeo90K is as follows:
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

  Vid4 ([Data Download](https://paddlegan.bj.bcebos.com/datasets/Vid4.zip)) is a commonly used test dataset for VSR, which contains 4 video segments.
  The structure of the processed Vid4 is as follows:
  ```
    PaddleGAN
      ├── data
          ├── Vid4
                ├── BDx4
                └── GT
              ...
  ```

  UDM10 ([Data Download](https://paddlegan.bj.bcebos.com/datasets/udm10_paddle.tar)) is a commonly used test dataset for VSR, which contains 10 video segments.
  The structure of the processed UDM10 is as follows:
  ```
    PaddleGAN
      ├── data
          ├── udm10
                ├── BDx4
                └── GT
              ...
  ```



### 1.2.2 Train/Test

  According to the number of channels, EDVR are divided into EDVR_L(128 channels) and EDVR_M (64 channels). Then, taking EDVR_M as an example, the model training and testing are introduced.

  The train of EDVR is generally divided into two stages. First, train EDVR without TSA module.

  The command to train and test edvr without TSA module is as follows:

  Train a model:
  ```
     python -u tools/main.py --config-file configs/edvr_m_wo_tsa.yaml
  ```

  Test the model:
  ```
     python tools/main.py --config-file configs/edvr_m_wo_tsa.yaml --evaluate-only --load ${PATH_OF_WEIGHT_WITHOUT_TSA}
  ```

  Then the weight of EDVR without TSA module is used as the initialization of edvr model to train the complete edvr model.

  The command to train and test edvr is as follows:

  Train a model:
  ```
     python -u tools/main.py --config-file configs/edvr_m_w_tsa.yaml --load ${PATH_OF_WEIGHT_WITHOUT_TSA}
  ```

  Test the model:
  ```
     python tools/main.py --config-file configs/edvr_m_w_tsa.yaml --evaluate-only --load ${PATH_OF_WEIGHT}
  ```

  To train or test other VSR model, you can find the config file of the corresponding VSR model in the ``PaddleGAN/configs``, then change the config file in the command to the config file of corresponding VSR model.


## 1.3 Results
The experimental results are evaluated on RGB channel.

The metrics are PSNR / SSIM.

VSR quantitative comparis on the test dataset REDS4 from REDS dataset
| Method | Paramete(M) | FLOPs(G) | REDS4 |
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

Deblur quantitative comparis on the test dataset REDS4 from REDS dataset
| Method | REDS4 |
|---|---|
| EDVR_L_wo_tsa_deblur  | 34.9587 / 0.9509 |
| EDVR_L_w_tsa_deblur  | 35.1473 / 0.9526 |

VSR quantitative comparis on the Vimeo90K, Vid4, UDM10
| Model | Vimeo90K | Vid4 | UDM10 |
|---|---|---|---|
| PP-MSVSR_vimeo90k_x4 |37.54/0.9499|28.13/0.8604|40.06/0.9699|

## 1.4 Model Download
| Method | Dataset | Download Link |
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



# References

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
