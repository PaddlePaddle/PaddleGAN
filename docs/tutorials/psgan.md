# PSGAN
## 1. PSGAN introduction
This paper is to address the makeup transfer task, which aims to transfer the makeup from a reference image to a source image. Existing methods have achieved promising progress in constrained scenarios, but transferring between images with large pose and expression differences is still challenging. To address these issues, we propose Pose and expression robust Spatial-aware GAN (PSGAN). It first utilizes Makeup Distill Network to disentangle the makeup of the reference image as two spatial-aware makeup matrices. Then, Attentive Makeup Morphing module is introduced to specify how the makeup of a pixel in the source image is morphed from the reference image. With the makeup matrices and the source image, Makeup Apply Network is used to perform makeup transfer.
![](../imgs/psgan_arc.png)

## 2. How to use
### 2.1 Test
Running the following command to complete the makeup transfer task. It will geneate the transfered image in the current path when the program running sucessfully.

```
python tools/ps_demo.py \  
  --config-file configs/makeup.yaml \
  --model_path /your/model/path \
  --source_path  /your/source/image/path  \
  --reference_dir /your/ref/image/path
```
** params: **
- config-file: PSGAN network configuration file, yaml format
- model_path: Saved model weight path
- source_path: Full path of the non-makeup image file, including the image file name
- reference_dir: Path of the make_up iamge file, don't including the image file name

### 2.2 Training
1. Downloading the original makeup transfer [data](https://pan.baidu.com/s/1ZF-DN9PvbBteOSfQodWnyw)(Password:rtdd) to the PaddleGAN folder, and uncompress it. Then downloading the landmarks [data](), and substituting the `landmarks` folder for the orinal `landmarks` folder. In addition, sustituting the `train_makeup.txt` file for the original `makeup.txt` file, and sustituting the `train_non-makeup.txt` file for the original `non-makeup.txt` file.
2. `python tools/main.py --config-file configs/makeup.yaml`

Notation: In train phase, the `isTrain` value in makeup.yaml file is `True`, but in test phase, its value should be modified as `False`.

## 3. Result
![](../imgs/makeup_shifter.png)
