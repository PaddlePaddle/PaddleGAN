# Fist order motion model

## First order motion model introduction

[First order motion model](https://arxiv.org/abs/2003.00196) is to complete the Image animation task, which consists of generating a video sequence so that an object in a source image is animated according to the motion of a driving video. The first order motion framework addresses this problem without using any annotation or prior information about the specific object to animate. Once trained on a set of videos depicting objects of the same category (e.g. faces, human bodies), this method can be applied to any object of this class. To achieve this, the innovative method decouple appearance and motion information using a self-supervised formulation. In addition, to support complex motions, it use a representation consisting of a set of learned keypoints along with their local affine transformations. A generator network models occlusions arising during target motions and combines the appearance extracted from the source image and the motion derived from the driving video.

<div align="center">
  <img src="../../imgs/fom_demo.png" width="500"/>
</div>

## How to use

Users can upload the prepared source image and driving video, then substitute the path of source image and driving video for the `source_image` and `driving_video` parameter in the following running command. It will geneate a video file named `result.mp4` in the `output` folder, which is the animated video file.

```
cd applications/
python -u tools/first-order-demo.py  \
     --driving_video ../docs/imgs/fom_dv.mp4 \
     --source_image ../docs/imgs/fom_source_image.png \
     --relative --adapt_scale
```

**params:**
- driving_video: driving video, the motion of the driving video is to be migrated.
- source_image: source_image, the image will be animated according to the motion of the driving video.
- relative: indicate whether the relative or absolute coordinates of the key points in the video are used in the program. It is recommended to use relative coordinates. If absolute coordinates are used, the characters will be distorted after animation.
- adapt_scale: adapt movement scale based on convex hull of keypoints.

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
