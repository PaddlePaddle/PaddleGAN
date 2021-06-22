
# 1 Video Super Resolution (VSR)

## 1.1 Principle

  Video super-resolution originates from image super-resolution, which aims to recover high-resolution (HR) images from one or more low resolution (LR) images. The difference between them is that the video is composed of multiple frames, so the video super-resolution usually uses the information between frames to repair. Here we provide the video super-resolution model [EDVR](https://arxiv.org/pdf/1905.02716.pdf).

  [EDVR](https://arxiv.org/pdf/1905.02716.pdf) wins the champions and outperforms the second place by a large margin in all four tracks in the NTIRE19 video restoration and enhancement challenges. The main difficulties of video super-resolution from two aspects: (1) how to align multiple frames given large motions, and (2) how to effectively fuse different frames with diverse motion and blur. First, to handle large motions, EDVR devise a Pyramid, Cascading and Deformable (PCD) alignment module, in which frame alignment is done at the feature level using deformable convolutions in a coarse-to-fine manner. Second, EDVR propose a Temporal and Spatial Attention (TSA) fusion module, in which attention is applied both temporally and spatially, so as to emphasize important features for subsequent restoration.



## 1.2 How to use  

### 1.2.1 Prepare Datasets

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


## 1.3 Results
The experimental results are evaluated on RGB channel.

The metrics are PSNR / SSIM.

| Method | REDS4 |
|---|---|
| EDVR_M_wo_tsa_SRx4  | 30.4429 / 0.8684 |
| EDVR_M_w_tsa_SRx4  | 30.5169 / 0.8699 |
| EDVR_L_wo_tsa_SRx4  | 30.8649 / 0.8761 |
| EDVR_L_w_tsa_SRx4  | 30.9336 / 0.8773 |


## 1.4 Model Download
| Method | Dataset | Download Link |
|---|---|---|
| EDVR_M_wo_tsa_SRx4  | REDS | [EDVR_M_wo_tsa_SRx4](https://paddlegan.bj.bcebos.com/models/EDVR_M_wo_tsa_SRx4.pdparams)
| EDVR_M_w_tsa_SRx4  | REDS | [EDVR_M_w_tsa_SRx4](https://paddlegan.bj.bcebos.com/models/EDVR_M_w_tsa_SRx4.pdparams)
| EDVR_L_wo_tsa_SRx4  | REDS | [EDVR_L_wo_tsa_SRx4](https://paddlegan.bj.bcebos.com/models/EDVR_L_wo_tsa_SRx4.pdparams)
| EDVR_L_w_tsa_SRx4  | REDS | [EDVR_L_w_tsa_SRx4](https://paddlegan.bj.bcebos.com/models/EDVR_L_w_tsa_SRx4.pdparams)





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
