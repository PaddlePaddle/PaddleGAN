# PSGAN
## 1. PSGAN introduction
This paper is to address the makeup transfer task, which aims to transfer the makeup from a reference image to a source image. Existing methods have achieved promising progress in constrained scenarios, but transferring between images with large pose and expression differences is still challenging. To address these issues, we propose Pose and expression robust Spatial-aware GAN (PSGAN). It first utilizes Makeup Distill Network to disentangle the makeup of the reference image as two spatial-aware makeup matrices. Then, Attentive Makeup Morphing module is introduced to specify how the makeup of a pixel in the source image is morphed from the reference image. With the makeup matrices and the source image, Makeup Apply Network is used to perform makeup transfer.
![](../imgs/psgan_arc.png)

## 2. How to use
### 2.1 Test
Running the following command to complete the makeup transfer task. It will geneate the transfered image in the current path when the program running sucessfully. We have provided two sample image file for the demo display.

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
1. Downloading [data]() to the PaddleGAN folder, and uncompress it
2. `python tools/main.py --config-file configs/makeup.yaml`

## 3. Result
![](../imgs/makeup_shifter.png)
