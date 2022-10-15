English | [Chinese](../../zh_CN/tutorials/nafnet.md)

## NAFNet：Simple Baselines for Image Restoration

## 1、Introduction

NAFNet proposes an ultra-simple baseline scheme, Baseline, which is not only computationally efficient but also outperforms the previous SOTA scheme; the resulting Baseline is further simplified to give NAFNet: the non-linear activation units are removed and the performance is further improved. The proposed solution achieves new SOTA performance for both SIDD noise reduction and GoPro deblurring tasks with a significant reduction in computational effort. The network design and features are shown in the figure below, using a UNet with skip connections as the overall architecture, modifying the Transformer module in the Restormer block and eliminating the activation function, adopting a simpler and more efficient simplegate design, and applying a simpler channel attention mechanism.

![NAFNet](https://ai-studio-static-online.cdn.bcebos.com/699b87449c7e495f8655ae5ac8bc0eb77bed4d9cd828451e8939ddbc5732a704)

For a more detailed introduction to the model, please refer to the original paper [Simple Baselines for Image Restoration](https://arxiv.org/pdf/2204.04676), PaddleGAN currently provides the weight of the denoising task.

## 2 How to use

### 2.1 Quick start

After installing PaddleGAN, you can run a command as follows to generate the restorated image.

```sh
python applications/tools/nafnet_denoising.py --images_path ${PATH_OF_IMAGE}
```
Where `PATH_OF_IMAGE` is the path of the image you need to denoise, or the path of the folder where the images is located. If you need to use your own model weights, run the following command, where `PATH_OF_MODEL` is the path to the model weights.

```sh
python applications/tools/nafnet_denoising.py --images_path ${PATH_OF_IMAGE}  --weight_path ${PATH_OF_MODEL}
```

### 2.2 Prepare dataset

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
Users can also use the [SIDD data](https://aistudio.baidu.com/aistudio/datasetdetail/149460) on AI studio, but need to rename the folders `input_crops` and `gt_crops` to `input` and ` target`

### 2.3 Training
An example is training to denoising. If you want to train for other tasks,If you want to train other tasks, you can change the dataset and modify the config file.

```sh
python -u tools/main.py --config-file configs/nafnet_denoising.yaml
```

### 2.4 Test

test model:
```sh
python tools/main.py --config-file configs/nafnet_denoising.yaml --evaluate-only --load ${PATH_OF_WEIGHT}
```

## 3 Results
Denoising
| model | dataset | PSNR/SSIM |
|---|---|---|
| NAFNet | SIDD Val |  43.1468 / 0.9563 |

## 4 Download

| model | link |
|---|---|
| NAFNet| [NAFNet_Denoising](https://paddlegan.bj.bcebos.com/models/NAFNet_Denoising.pdparams) |

# References

- [Simple Baselines for Image Restoration](https://arxiv.org/pdf/2204.04676)

```
@article{chen_simple_nodate,
	title = {Simple {Baselines} for {Image} {Restoration}},
	abstract = {Although there have been significant advances in the field of image restoration recently, the system complexity of the state-of-the-art (SOTA) methods is increasing as well, which may hinder the convenient analysis and comparison of methods. In this paper, we propose a simple baseline that exceeds the SOTA methods and is computationally efficient. To further simplify the baseline, we reveal that the nonlinear activation functions, e.g. Sigmoid, ReLU, GELU, Softmax, etc. are not necessary: they could be replaced by multiplication or removed. Thus, we derive a Nonlinear Activation Free Network, namely NAFNet, from the baseline. SOTA results are achieved on various challenging benchmarks, e.g. 33.69 dB PSNR on GoPro (for image deblurring), exceeding the previous SOTA 0.38 dB with only 8.4\% of its computational costs; 40.30 dB PSNR on SIDD (for image denoising), exceeding the previous SOTA 0.28 dB with less than half of its computational costs. The code and the pretrained models will be released at github.com/megvii-research/NAFNet.},
	language = {en},
	author = {Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
	pages = {17}
}
```




