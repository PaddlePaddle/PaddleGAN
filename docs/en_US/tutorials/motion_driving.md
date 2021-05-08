# First Order Motion model

## First Order Motion model introduction

[First order motion model](https://arxiv.org/abs/2003.00196) is to complete the Image animation task, which consists of generating a video sequence so that an object in a source image is animated according to the motion of a driving video. The first order motion framework addresses this problem without using any annotation or prior information about the specific object to animate. Once trained on a set of videos depicting objects of the same category (e.g. faces, human bodies), this method can be applied to any object of this class. To achieve this, the innovative method decouple appearance and motion information using a self-supervised formulation. In addition, to support complex motions, it use a representation consisting of a set of learned keypoints along with their local affine transformations. A generator network models occlusions arising during target motions and combines the appearance extracted from the source image and the motion derived from the driving video.

<div align="center">
  <img src="../../imgs/fom_demo.png" width="500"/>
</div>
## Multi-Faces swapping

For photoes with multiple faces, we first detect all of the faces,  then do facial expression transfer for each face, and finally put those faces back to the original photo to generate a complete new video.

Specific technical steps are shown below:

1. Use the S3FD model to detect the faces of a photo
2. Use the First Order Motion model to do the facial expression transfer of each face
3. Put those "new" generated faces back to the original photo

At the same time, specifically for face related work, PaddleGAN provides a ["faceutils" tool](https://github.com/PaddlePaddle/PaddleGAN/tree/develop/ppgan/faceutils), including face detection, face segmentation models and more.

## How to use
### 1 Test for Face
Users can upload the prepared source image and driving video, then substitute the path of source image and driving video for the `source_image` and `driving_video` parameter in the following running command. It will geneate a video file named `result.mp4` in the `output` folder, which is the animated video file.

Note: for photoes with multiple faces, the longer the distances between faces, the better the result quality you can get.  

- single face:
```
cd applications/
python -u tools/first-order-demo.py  \
     --driving_video ../docs/imgs/fom_dv.mp4 \
     --source_image ../docs/imgs/fom_source_image.png \
     --ratio 0.4 \
     --relative --adapt_scale
```

- multi face：
```
cd applications/
python -u tools/first-order-demo.py  \
     --driving_video ../docs/imgs/fom_dv.mp4 \
     --source_image ../docs/imgs/fom_source_image_multi_person.png \
     --ratio 0.4 \
     --relative --adapt_scale \
     --multi_person


**params:**
- driving_video: driving video, the motion of the driving video is to be migrated.
- source_image: source_image, support single people and multi-person in the image, the image will be animated according to the motion of the driving video.
- relative: indicate whether the relative or absolute coordinates of the key points in the video are used in the program. It is recommended to use relative coordinates. If absolute coordinates are used, the characters will be distorted after animation.
- adapt_scale: adapt movement scale based on convex hull of keypoints.
- ratio: The pasted face percentage of generated image, this parameter should be adjusted in the case of multi-person image in which the adjacent faces are close. The defualt value is 0.4 and the range is [0.4, 0.5].
- multi_person: There are multi faces in the images. Default means only one face in the image

### 2 Training
**Datasets:**
- fashion See[here](https://vision.cs.ubc.ca/datasets/fashion/)
- VoxCeleb See[here](https://github.com/AliaksandrSiarohin/video-preprocessing)

**params:**
- dataset_name.yaml: Create a config of your own dataset

- For single GPU:
```
export CUDA_VISIBLE_DEVICES=0
python tools/main.py --config-file configs/dataset_name.yaml
```
- For multiple GPUs:
```
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch \
    tools/main.py \
    --config-file configs/dataset_name.yaml

```

**Example:**
- For single GPU:
```
export CUDA_VISIBLE_DEVICES=0
python tools/main.py --config-file configs/firstorder_fashion.yaml \
```
- For multiple GPUs:
```
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch \
    tools/main.py \
    --config-file configs/firstorder_fashion.yaml \
```


**Online Tutorial running in AI Studio:**

* **Multi-faces swapping: https://aistudio.baidu.com/aistudio/projectdetail/1603391**
* **Single face swapping: https://aistudio.baidu.com/aistudio/projectdetail/1586056**

## Animation results

![](../../imgs/first_order.gif)


## Reference

```
@InProceedings{Siarohin_2019_NeurIPS,
  author={Siarohin, Aliaksandr and Lathuilière, Stéphane and Tulyakov, Sergey and Ricci, Elisa and Sebe, Nicu},
  title={First Order Motion Model for Image Animation},
  booktitle = {Conference on Neural Information Processing Systems (NeurIPS)},
  month = {December},
  year = {2019}
}
```
