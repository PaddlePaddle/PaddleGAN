# SinGAN

## Introduction

SinGAN is a novel unconditional* generative model that is trained using a single image. Traditionally, GANs have been trained on class-specific datasets and capture common features among images of the same class. SinGAN, on the other hand, learns from the overlapping patches at multiple scales of a particular image and learns its internal statistics. Once trained, SinGAN can produce assorted high-quality images of arbitrary sizes and aspect ratios that semantically resemble the training image but contain new object configurations and structures.

** An unconditional GAN creates samples purely from randomized input, while a conditional GAN generates samples based on a "class label" that controls the type of image generated.*

## Usage

### About Config Files

We provide 4 config files for SinGAN model:

- `singan_universal.yaml`
- `singan_sr.yaml`
- `singan_animation.yaml`
- `singan_finetune.yaml`

Among them, `singan_universal.yaml` is a config file suit for all tasks, `singan_sr.yaml` is a config file for super resolution recommended by the author, `singan_animation.yaml` is a config file for animation recommended by the author. Results showed in this document were trained with `singan_universal.yaml`. For *Paint to Image*, we will get better results by finetuning with `singan_finetune.yaml` after training with `singan_universal.yaml`.

### Train

Start training:

```bash
python tools/main.py -c configs/singan_universal.yaml \
                     -o model.train_image=train_image.png
```

Finetune for "Paint2Image":

```bash
python tools/main.py -c configs/singan_finetune.yaml \
                     -o model.train_image=train_image.png \
                     --load weight_saved_in_training.pdparams
```

### Evaluation
Running following command, a random image will be generated. It should be noted that `train_image.png` ought to be in directory `data/singan`, or you can modify the value of `dataset.test.dataroot` in config file manually. Besides, this directory must contain only one image, which is `train_image.png`.
```bash
python tools/main.py -c configs/singan_universal.yaml \
                     -o model.train_image=train_image.png \
                     --load weight_saved_in_training.pdparams \
                     --evaluate-only
```

### Extract Weight for Generator

After training, we need use ``tools/extract_weight.py`` to extract weight of generator from training model which includes both generator and discriminator. Then we can use `applications/tools/singan.py` to achieve diverse application of SinGAN.

```bash
python tools/extract_weight.py weight_saved_in_training.pdparams --net-name netG --output weight_of_generator.pdparams
```

### Inference and Result

*Attention: to use pretrained model, you can replace `--weight_path weight_of_generator.pdparams` in the following commands by `--pretrained_model <model>`, where `<model>` can be `trees`, `stone`, `mountains`, `birds` or `lightning`.*

#### Random Sample

```bash
python applications/tools/singan.py \
       --weight_path weight_of_generator.pdparams \
       --mode random_sample \
       --scale_v 1 \ # vertical scale
       --scale_h 1 \ # horizontal scale
       --n_row 2 \
       --n_col 2
```

|training image|result|
| ---- | ---- |
|![birds](https://user-images.githubusercontent.com/91609464/153211448-2614407b-a30b-467c-b1e5-7db88ff2ca74.png)|![birds-random_sample](https://user-images.githubusercontent.com/91609464/153211573-1af108ba-ad42-438a-94a9-e8f8f3e091eb.png)|

#### Editing & Harmonization

```bash
python applications/tools/singan.py \
       --weight_path weight_of_generator.pdparams \
       --mode editing \ # or harmonization
       --ref_image editing_image.png \
       --mask_image mask_of_editing.png \
       --generate_start_scale 2
```


|training image|editing image|mask of editing|result|
|----|----|----|----|
|![stone](https://user-images.githubusercontent.com/91609464/153211778-bb94d29d-a2b4-4d04-9900-89b20ae90b90.png)|![stone-edit](https://user-images.githubusercontent.com/91609464/153211867-df3d9035-d320-45ec-8043-488e9da49bff.png)|![stone-edit-mask](https://user-images.githubusercontent.com/91609464/153212047-9620f73c-58d9-48ed-9af7-a11470ad49c8.png)|![stone-edit-mask-result](https://user-images.githubusercontent.com/91609464/153211942-e0e639c2-3ea6-4ade-852b-73757b0bbab0.png)|

#### Super Resolution

```bash
python applications/tools/singan.py \
       --weight_path weight_of_generator.pdparams \
       --mode sr \
       --ref_image image_to_sr.png \
       --sr_factor 4
```
|training image|result|
| ---- | ---- |
|![mountains](https://user-images.githubusercontent.com/91609464/153212146-efbbbbd6-e045-477a-87ae-10f121341060.png)|![sr](https://user-images.githubusercontent.com/91609464/153212176-530b7075-e72b-4c05-ad3e-2f2cdfc76dea.png)|


#### Animation

```bash
python applications/tools/singan.py \
       --weight_path weight_of_generator.pdparams \
       --mode animation \
       --animation_alpha 0.6 \ # this parameter determines how close the frames of the sequence remain to the training image
       --animation_beta 0.7 \ # this parameter controls the smoothness and rate of change in the generated clip
       --animation_frames 20 \ # frames of animation
       --animation_duration 0.1	# duration of each frame
```

|training image|animation|
| ---- | ---- |
|![lightning](https://user-images.githubusercontent.com/91609464/153212291-6f8976bd-e873-423e-ab62-77997df2df7a.png)|![animation](https://user-images.githubusercontent.com/91609464/153212372-0543e6d6-5842-472b-af50-8b22670270ae.gif)|


#### Paint to Image
```bash
python applications/tools/singan.py \
       --weight_path weight_of_generator.pdparams \
       --mode paint2image \
       --ref_image paint.png \
       --generate_start_scale 2
```
|training image|paint|result|result after finetune|
|----|----|----|----|
|![trees](https://user-images.githubusercontent.com/91609464/153212536-0bb6489d-d488-49e0-a6b5-90ef578c9e4f.png)|![trees-paint](https://user-images.githubusercontent.com/91609464/153212511-ef2c6bea-1f8c-4685-951b-8db589414dfe.png)|![trees-paint2image](https://user-images.githubusercontent.com/91609464/153212531-c080c705-fd58-4ade-aac6-e2134838a75f.png)|![trees-paint2image-finetuned](https://user-images.githubusercontent.com/91609464/153212529-51d8d29b-6b58-4f29-8792-4b2b04f9266e.png)|

## Reference

```
@misc{shaham2019singan,
      title={SinGAN: Learning a Generative Model from a Single Natural Image}, 
      author={Tamar Rott Shaham and Tali Dekel and Tomer Michaeli},
      year={2019},
      eprint={1905.01164},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

