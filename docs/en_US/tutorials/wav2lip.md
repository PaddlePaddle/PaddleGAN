# Lip-syncing

## 1. Lip-syncing introduction

This work address the problem of lip-syncing a talking face video of an arbitray identity to match a target speech segment. Current works excel at producing accurate lip movements on a static image or on videosof specific people seen during the training phase. Wav2lip tackle this problem by learning from a powerful lip-sync discriminator, and the result show that the lip-sync accuracy of the generated videos using Wav2Lip model is almost as good as real synced videos.
## 2. How to use

### 2.1 Test
The pretrained model can be downloaded from [here](https://paddlegan.bj.bcebos.com/models/wav2lip_hq.pdparams)
Runing the following command to complete the lip-syning task. The output is the synced videos.

```
cd applications
python tools/wav2lip.py \
    --face ../docs/imgs/mona7s.mp4 \
    --audio ../docs/imgs/guangquan.m4a \
    --outfile pp_guangquan_mona7s.mp4 \
    --face_enhancement
```

**params:**

- face: path of the input image or video file including faces.
- audio: path of the input audio file, format can be `.wav`， `.mp3`, `.m4a`. It can be any file supported by `FFMPEG` containing audio data.
- outfile: result video of wav2lip
- face_enhancement: enhance the face, default is False

### 2.2 Training
1. Our model are trained on LRS2. See [here](https://github.com/Rudrabha/Wav2Lip#training-on-datasets-other-than-lrs2) for a few suggestions regarding training on other datasets.

Preprocessed LRS2 dataset folder structure should be like:
```
preprocessed_root (lrs2_preprocessed)
├── list of folders
|    ├── Folders with five-digit numbered video IDs
|    │   ├── *.jpg
|    │   ├── audio.wav
```
Place the LRS2 filelists(train, val, test) `.txt` files in the `filelists/` folder.

2. You can eigher train the model without the additional visual quality discriminator or use the discriminator. For the former, run:
- For single GPU:
```
export CUDA_VISIBLE_DEVICES=0
python tools/main.py --config-file configs/wav2lip.yaml
```

- For multiple GPUs:
```
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch \
    tools/main.py \
    --config-file configs/wav2lip.yaml \

```
For the latter, run:
- For single GPU:
```
export CUDA_VISIBLE_DEVICES=0
python tools/main.py --config-file configs/wav2lip_hq.yaml
```
- For multiple GPUs:
```
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch \
    tools/main.py \
    --config-file configs/wav2lip_hq.yaml \

```

### 2.3 Model

Model|Dataset|BatchSize|Inference speed|Download
---|:--:|:--:|:--:|:--:
wa2lip_hq|LRS2| 1 | 0.2853s/image (GPU:P40) | [model](https://paddlegan.bj.bcebos.com/models/wav2lip_hq.pdparams)

## Results
![](../../imgs/mona.gif)

### 4. Reference

```
@inproceedings{10.1145/3394171.3413532,
author = {Prajwal, K R and Mukhopadhyay, Rudrabha and Namboodiri, Vinay P. and Jawahar, C.V.},
title = {A Lip Sync Expert Is All You Need for Speech to Lip Generation In the Wild},
year = {2020},
isbn = {9781450379885},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3394171.3413532},
doi = {10.1145/3394171.3413532},
booktitle = {Proceedings of the 28th ACM International Conference on Multimedia},
pages = {484–492},
numpages = {9},
keywords = {lip sync, talking face generation, video generation},
location = {Seattle, WA, USA},
series = {MM '20}
}
```
