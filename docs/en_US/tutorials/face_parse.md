# Face Parsing

## 1. Face parsing introduction

Face parsing address the task that how to parse facial components from face images. We utiize BiseNet to handle this problem and focus on computing the pixel-wise label map of a face image. It is useful for a variety of tasks, including recognition, animation, and synthesis.  This application is now working in our makeup transfer model.

## 2. How to use

### 2.1 Test

Runing the following command to complete the face parsing task. The output results will be the segmanted face components mask for the input image.

```
cd applications
python tools/face_parse.py --input_image ../docs/imgs/face.png
```

**params:**

- input_image: path of the input face image

## Results
![](../../imgs/face_parse_out.png)

### 4. Reference

```
@misc{yu2018bisenet,
      title={BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation},
      author={Changqian Yu and Jingbo Wang and Chao Peng and Changxin Gao and Gang Yu and Nong Sang},
      year={2018},
      eprint={1808.00897},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
