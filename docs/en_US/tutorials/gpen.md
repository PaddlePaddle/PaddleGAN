English | [Chinese](../../zh_CN/tutorials/gpen.md)

## GPEN Blind Face Restoration Model


## 1、Introduction

The GPEN model is a blind face restoration model. The author embeds the decoder of StyleGAN V2 proposed by the previous model as the decoder of GPEN; reconstructs a simple encoder with DNN to provide input for the decoder. In this way, while retaining the excellent performance of the StyleGAN V2 decoder, the function of the model is changed from image style conversion to blind face restoration. The overall structure of the model is shown in the following figure:

![img](https://user-images.githubusercontent.com/23252220/168281766-a0972bd3-243e-4fc7-baa5-e458ef0946ce.jpg)

For a more detailed introduction to the model, and refer to the repo, you can view the following AI Studio project [link]([GPEN Blind Face Repair Model Reproduction - Paddle AI Studio (baidu.com)](https://aistudio.baidu.com/ The latest version of aistudio/projectdetail/3936241?contributionType=1)).




## 2、Ready to work

### 2.1 Dataset Preparation

The GPEN model training set is the classic FFHQ face data set, with a total of 70,000 high-resolution 1024 x 1024 high-resolution face pictures, and the test set is the CELEBA-HQ data set, with a total of 2,000 high-resolution face pictures. For details, please refer to **Dataset URL:** [FFHQ](https://github.com/NVlabs/ffhq-dataset), [CELEBA-HQ](https://github.com/tkarras/progressive_growing_of_gans). The specific download links are given below:

**Original dataset download address:**

**FFHQ ：**           https://drive.google.com/drive/folders/1tZUcXDBeOibC6jcMCtgRRz67pzrAHeHL?usp=drive_open

**CELEBA-HQ：** https://drive.google.com/drive/folders/0B4qLcYyJmiz0TXY1NG02bzZVRGs?resourcekey=0-arAVTUfW9KRhN-irJchVKQ&usp=sharing



Since the original FFHQ dataset is too large, you can also download the 256-resolution FFHQ dataset from the following link:

https://paddlegan.bj.bcebos.com/datasets/images256x256.tar



**After downloading, the file organization is as follows**

```
|-- data/GPEN
	|-- ffhq/images256x256/
		|-- 00000
			|-- 00000.png
			|-- 00001.png
			|-- ......
			|-- 00999.png
		|-- 01000
			|-- ......
		|-- ......
		|-- 69000
            |-- ......
                |-- 69999.png
	|-- test
		|-- 2000张png图片
```

Please modify the dataroot parameters of dataset train and test in the configs/gpen_256_ffhq.yaml configuration file to your training set and test set path.



### 2.2 Model preparation

**Model parameter file and training log download address:**

link：https://paddlegan.bj.bcebos.com/models/gpen.zip


Download the model parameters and test images from the link and put them in the data/ folder in the project root directory. The specific file structure is as follows:


```
data/gpen/weights
    |-- model_ir_se50.pdparams
    |-- weight_pretrain.pdparams  
data/gpen/lite_data
```



## 3、Start using

### 3.1 model training

Enter the following code in the console to start training：

 ```shell
 python tools/main.py -c configs/gpen_256_ffhq.yaml
 ```

The model only supports single-card training.

Model training needs to use paddle2.3 and above, and wait for paddle to implement the second-order operator related functions of elementwise_pow. The paddle2.2.2 version can run normally, but the model cannot be successfully trained because some loss functions will calculate the wrong gradient. . If an error is reported during training, training is not supported for the time being. You can skip the training part and directly use the provided model parameters for testing. Model evaluation and testing can use paddle2.2.2 and above.



### 3.2 Model evaluation

When evaluating the model, enter the following code in the console, using the downloaded model parameters mentioned above:

 ```shell
python tools/main.py -c configs/gpen_256_ffhq.yaml -o dataset.test.amount=2000 --load data/gpen/weights/weight_pretrain.pdparams --evaluate-only
 ```

If you want to test on your own provided model, please modify the path after --load .



### 3.3 Model prediction

#### 3.3.1 Export generator weights

After training, you need to use ``tools/extract_weight.py`` to extract the weights of the generator from the trained model (including the generator and discriminator) for inference to `applications/tools/gpen.py` to achieve Various applications of the GPEN model. Enter the following command to extract the weights of the generator:

```bash
python tools/extract_weight.py data/gpen/weights/weight_pretrain.pdparams --net-name g_ema --output data/gpen/weights/g_ema.pdparams
```



#### 3.3.2 Process a single image

After extracting the weights of the generator, enter the following command to test the images under the --test_img path. Modifying the --seed parameter can generate different degraded images to show richer effects. You can modify the path after --test_img to any image you want to test. If no weight is provided after the --weight_path parameter, the trained model weights will be automatically downloaded for testing.

```bash
python applications/tools/gpen.py --test_img data/gpen/lite_data/15006.png --seed=100 --weight_path data/gpen/weights/g_ema.pdparams --model_type gpen-ffhq-256
```

The following are the sample images and the corresponding inpainted images, from left to right, the degraded image, the generated image, and the original clear image:

<p align='center'>
<img src="https://user-images.githubusercontent.com/23252220/168281788-39c08e86-2dc3-487f-987d-93489934c14c.png" height="256px" width='768px' >
An example output is as follows:


```
result saved in : output_dir/gpen_predict.png
        FID: 92.11730631094356
        PSNR:19.014782083825743
```



## 4. Tipc

### 4.1 Export the inference model

```bash
python tools/export_model.py -c configs/gpen_256_ffhq.yaml --inputs_size=1,3,256,256 --load data/gpen/weights/weight_pretrain.pdparams
```

The above command will generate the model structure file `gpenmodel_g_ema.pdmodel` and model weight files `gpenmodel_g_ema.pdiparams` and `gpenmodel_g_ema.pdiparams.info` files required for prediction, which are stored in the `inference_model/` directory. You can also modify the parameters after --load to the model parameter file you want to test.



### 4.2 Inference with a prediction engine

```bash
python tools/inference.py --model_type GPEN --seed 100 -c configs/gpen_256_ffhq.yaml -o dataset.test.dataroot="./data/gpen/lite_data/" --output_path test_tipc/output/ --model_path inference_model/gpenmodel_g_ema
```

At the end of the inference, the repaired image generated by the model will be saved in the test_tipc/output/GPEN directory by default, and the FID value obtained by the test will be output in test_tipc/output/GPEN/metric.txt.


The default output is as follows:

```
Metric fid: 187.0158
```

Note: Since the operation of degrading high-definition pictures has a certain degree of randomness, the results of each test will be different. In order to ensure that the test results are consistent, here I fixed the random seed, so that the same degradation operation is performed on the image for each test.



### 4.3 Call the script to complete the training and push test in two steps

To invoke the `lite_train_lite_infer` mode of the foot test base training prediction function, run:

```shell
# Corrected format of sh file
sed -i 's/\r//' test_tipc/prepare.sh
sed -i 's/\r//' test_tipc/test_train_inference_python.sh
sed -i 's/\r//' test_tipc/common_func.sh
# prepare data
bash test_tipc/prepare.sh ./test_tipc/configs/GPEN/train_infer_python.txt 'lite_train_lite_infer'
# run the test
bash test_tipc/test_train_inference_python.sh ./test_tipc/configs/GPEN/train_infer_python.txt 'lite_train_lite_infer'
```



## 5、References

```
@misc{2021GAN,
      title={GAN Prior Embedded Network for Blind Face Restoration in the Wild},
      author={ Yang, T.  and  Ren, P.  and  Xie, X.  and  Zhang, L. },
      year={2021},
      archivePrefix={CVPR},
      primaryClass={cs.CV}
}
```

