# AOT GAN

## 1 Principle

  The Aggregated COntextual-Transformation GAN (AOT-GAN) is for high-resolution image inpainting.The AOT blocks aggregate contextual
transformations from various receptive fields, allowing to capture both informative distant image contexts and rich patterns of interest
for context reasoning.

![](https://ai-studio-static-online.cdn.bcebos.com/c3b71d7f28ce4906aa7cccb10ed09ae5e317513b6dbd471aa5cca8144a7fd593)

**Paper:** [Aggregated Contextual Transformations for High-Resolution Image Inpainting](https://paperswithcode.com/paper/aggregated-contextual-transformations-for)

**Official Repo:** [https://github.com/megvii-research/NAFNet](https://github.com/megvii-research/NAFNet)


## 2 How to use 

### 2.1 Prediction

Download pretrained generator weights from: (https://paddlegan.bj.bcebos.com/models/AotGan_g.pdparams)

```
python applications/tools/aotgan.py \
	--input_image_path data/aotgan/armani1.jpg \
	--input_mask_path data/aotgan/armani1.png \
	--weight_path test/aotgan/g.pdparams \
	--output_path output_dir/armani_pred.jpg \
	--config-file configs/aotgan.yaml
```
Parameters:
* input_image_path：input image
* input_mask_path：input mask
* weight_path：pretrained generator weights
* output_path：predicted image
* config-file：yaml file,same with the training process

AI Studio Project:(https://aistudio.baidu.com/aistudio/datasetdetail/165081)

### 2.2 Train

Data Preparation:

The pretained model uses 'Place365Standard' and 'NVIDIA Irregular Mask' as its training datasets. You can download then from ([Place365Standard](http://places2.csail.mit.edu/download.html)) and ([NVIDIA Irregular Mask Dataset](https://nv-adlr.github.io/publication/partialconv-inpainting)).
  
```
└─data
    └─aotgan
        ├─train_img
        ├─train_mask
        ├─val_img
        └─val_mask
```
Train(Single Card):

`python -u tools/main.py --config-file configs/aotgan.yaml`

Train(Mult-Card):

```
!python -m paddle.distributed.launch \
    tools/main.py \
    --config-file configs/photopen.yaml \
    -o dataset.train.batch_size=6
```
Train(continue):

```
python -u tools/main.py \
	--config-file configs/aotgan.yaml \
	--resume  output_dir/[path_to_checkpoint]/iter_[iternumber]_checkpoint.pdparams
```

# Results

On Places365-Val Dataset

|  mask   | PSNR  | SSIM  | download  |
|  ----  | ----  | ----  | ----  |
|  20-30%   | 26.04001  | 0.89011  | [download](https://paddlegan.bj.bcebos.com/models/AotGan_g.pdparams)  |

# References

@inproceedings{yan2021agg,
  author = {Zeng, Yanhong and Fu, Jianlong and Chao, Hongyang and Guo, Baining},
  title = {Aggregated Contextual Transformations for High-Resolution Image Inpainting},
  booktitle = {Arxiv},
  pages={-},
  year = {2020}
}
