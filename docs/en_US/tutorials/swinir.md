English | [Chinese](../../zh_CN/tutorials/swinir.md)

## SwinIR Strong Baseline Model for Image Restoration Based on Swin Transformer

## 1、Introduction

The structure of SwinIR is relatively simple. If you have seen Swin-Transformer, there is no difficulty. The authors introduce the Swin-T structure for low-level vision tasks, including image super-resolution reconstruction, image denoising, and image compression artifact removal. The SwinIR network consists of a shallow feature extraction module, a deep feature extraction module and a reconstruction module. The reconstruction module uses different structures for different tasks. Shallow feature extraction is a 3×3 convolutional layer. Deep feature extraction is composed of k RSTB blocks and a convolutional layer plus residual connections. Each RSTB (Res-Swin-Transformer-Block) consists of L STLs and a layer of convolution plus residual connections. The structure of the model is shown in the following figure:

![](https://ai-studio-static-online.cdn.bcebos.com/b550e84915634951af756a545c643c815001be73372248b0b5179fd1652ae003)

For a more detailed introduction to the model, please refer to the original paper [SwinIR: Image Restoration Using Swin Transformer](https://arxiv.org/pdf/2108.10257.pdf), PaddleGAN currently provides the weight of the denoising task.

## 2 How to use

### 2.1 Quick start

After installing PaddleGAN, you can run a command as follows to generate the restorated image.

```sh
python applications/tools/swinir_denoising.py --images_path ${PATH_OF_IMAGE}
```
Where `PATH_OF_IMAGE` is the path of the image you need to denoise, or the path of the folder where the images is located.

### 2.2 Prepare dataset

#### Train Dataset

[DIV2K](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar) (800 training images) + [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) (2650 images) + [BSD500](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz) (400 training&testing images) + [WED](http://ivc.uwaterloo.ca/database/WaterlooExploration/exploration_database_and_code.rar)(4744 images)

The data that has been sorted out: put it in [Ai Studio](https://aistudio.baidu.com/aistudio/datasetdetail/149405).

The training data is placed under: `data/trainsets/trainH`

#### Test Dataset

The test data is CBSD68: put it in [Ai Studio](https://aistudio.baidu.com/aistudio/datasetdetail/147756).

Extract to: `data/triansets/CBSD68`

### 2.3 Training
An example is training to denoising. If you want to train for other tasks,If you want to train other tasks, you can change the dataset and modify the config file.

```sh
python -u tools/main.py --config-file configs/swinir_denoising.yaml
```

### 2.4 Test

test model:
```sh
python tools/main.py --config-file configs/swinir_denoising.yaml --evaluate-only --load ${PATH_OF_WEIGHT}
```

## 3 Results
Denoising
| model | dataset | PSNR/SSIM |
|---|---|---|
| SwinIR | CBSD68 |  36.0819 / 0.9464 |

## 4 Download

| model | link |
|---|---|
| SwinIR| [SwinIR_Denoising](https://paddlegan.bj.bcebos.com/models/SwinIR_Denoising.pdparams) |

# References

- [SwinIR: Image Restoration Using Swin Transformer](https://arxiv.org/pdf/2108.10257.pdf)

```
@article{liang2021swinir,
    title={SwinIR: Image Restoration Using Swin Transformer},
    author={Liang, Jingyun and Cao, Jiezhang and Sun, Guolei and Zhang, Kai and Van Gool, Luc and Timofte, Radu},
    journal={arXiv preprint arXiv:2108.10257},
    year={2021}
}
```




