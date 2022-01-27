# MPR_Net

## 1 Introduction

[MPR_Net](https://arxiv.org/abs/2102.02808) is an image restoration method published in CVPR2021. Image restoration tasks demand a complex balance between spatial details and high-level contextualized information while recovering images. MPR_Net propose a novel synergistic design that can optimally balance these competing goals. The main proposal is a multi-stage architecture, that progressively learns restoration functions for the degraded inputs, thereby breaking down the overall recovery process into more manageable steps. Specifically, the model first learns the contextualized features using encoder-decoder architectures and later combines them with a high-resolution branch that retains local information. At each stage, MPR_Net introduce a novel per-pixel adaptive design that leverages in-situ supervised attention to reweight the local features. A key ingredient in such a multi-stage architecture is the information exchange between different stages. To this end, MPR_Net propose a two-faceted approach where the information is not only exchanged sequentially from early to late stages, but lateral connections between feature processing blocks also exist to avoid any loss of information. The resulting tightly interlinked multi-stage architecture, named as MPRNet, delivers strong performance gains on ten datasets across a range of tasks including image deraining, deblurring, and denoising.

## 2 How to use

### 2.1 Quick start

After installing PaddleGAN, you can run python code as follows to generate the restorated image. Where the `task` is the type of restoration method, you can chose in `Deblurring`、`Denoising` and `Deraining`, and `PATH_OF_IMAGE`is your image path.

```python
from ppgan.apps import MPRPredictor
predictor = MPRPredictor(task='Deblurring')
predictor.run(PATH_OF_IMAGE)
```

Or run such a command to get the same result:

```sh
python applications/tools/mprnet.py --input_image ${PATH_OF_IMAGE} --task Deblurring
```
Where the `task` is the type of restoration method, you can chose in `Deblurring`、`Denoising` and `Deraining`, and `PATH_OF_IMAGE`is your image path.

### 2.1 Prepare dataset

The Deblurring training datasets is GoPro. The GoPro datasets used for deblurring consists of 3214 blurred images with a size of 1,280×720. These images are divided into 2103 training images and 1111 test images. It can be downloaded from [here](https://drive.google.com/file/d/1H0PIXvJH4c40pk7ou6nAwoxuR4Qh_Sa2/view?usp=sharing).
After downloading, decompress it to the data directory. After decompression, the structure of `GoProdataset` is as following:

```sh
GoPro
├── train
│   ├── input
│   └── target
└── test
    ├── input
    └── target

```

The Denoising training datasets is SIDD, an image denoising datasets, containing 30,000 noisy images from 10 different lighting conditions, which can be downloaded from [training datasets](https://www.eecs.yorku.ca/~kamel/sidd/dataset.php) and [Test datasets](https://drive.google.com/drive/folders/1S44fHXaVxAYW3KLNxK41NYCnyX9S79su).
After downloading, decompress it to the data directory. After decompression, the structure of `SIDDdataset` is as following:

```sh
SIDD
├── train
│   ├── input
│   └── target
└── val
    ├── input
    └── target

```

Deraining training datasets is Synthetic Rain Datasets, which consists of 13,712 clean rain image pairs collected from multiple datasets (Rain14000, Rain1800, Rain800, Rain12), which can be downloaded from [training datasets](https://drive.google.com/drive/folders/1Hnnlc5kI0v9_BtfMytC2LR5VpLAFZtVe) and [Test datasets](https://drive.google.com/drive/folders/1PDWggNh8ylevFmrjo-JEvlmqsDlWWvZs).
After downloading, decompress it to the data directory. After decompression, the structure of `Synthetic_Rain_Datasets` is as following:

```sh
Synthetic_Rain_Datasets
├── train
│   ├── input
│   └── target
└── test
    ├── Test100
    ├── Rain100H
    ├── Rain100L
    ├── Test1200
    └── Test2800

```

### 2.2 Training
  An example is training to deblur. If you want to train for other tasks, you can replace the config file.

  ```sh
  python -u tools/main.py --config-file configs/mprnet_deblurring.yaml
  ```

### 2.3 Test

test model:
```sh
python tools/main.py --config-file configs/mprnet_deblurring.yaml --evaluate-only --load ${PATH_OF_WEIGHT}
```

## 3 Results
Deblurring
| model | dataset | PSNR/SSIM |
|---|---|---|
| MPRNet | GoPro | 33.4360/0.9410 |

Denoising
| model | dataset | PSNR/SSIM |
|---|---|---|
| MPRNet | SIDD |  43.6100 / 0.9586 |

Deraining
| model | dataset | PSNR/SSIM |
|---|---|---|
| MPRNet | Rain100L | 36.2848 / 0.9651 |

## 4 Download

| model | link |
|---|---|
| MPR_Deblurring | [MPR_Deblurring](https://paddlegan.bj.bcebos.com/models/MPR_Deblurring.pdparams) |
| MPR_Denoising | [MPR_Denoising](https://paddlegan.bj.bcebos.com/models/MPR_Denoising.pdparams) |
| MPR_Deraining | [MPR_Deraining](https://paddlegan.bj.bcebos.com/models/MPR_Deraining.pdparams) |



# References

- [Multi-Stage Progressive Image Restoration](https://arxiv.org/abs/2102.02808)

  ```
  @inproceedings{Kim2020U-GAT-IT:,
    title={Multi-Stage Progressive Image Restoration},
    author={Syed Waqas Zamir and Aditya Arora and Salman Khan and Munawar Hayat and Fahad Shahbaz Khan and Ming-Hsuan Yang and Ling Shao},
    booktitle={CVPR},
    year={2021}
  }
  ```
