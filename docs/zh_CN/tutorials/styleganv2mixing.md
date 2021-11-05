# StyleGAN V2 Mixing 模块

## StyleGAN V2 Mixing 原理

StyleGAN V2 的任务是使用风格向量进行image generation，而Mixing模块则是利用其风格向量实现两张生成图像不同层次不同比例的混合

## 使用方法

### 混合

用户使用如下命令中进行混合：

```
cd applications/
python -u tools/styleganv2mixing.py \
       --latent1 <替换为第一个风格向量的路径> \
       --latent2 <替换为第二个风格向量的路径> \
       --weights \
                 0.5 0.5 0.5 0.5 0.5 0.5 \
                 0.5 0.5 0.5 0.5 0.5 0.5 \
                 0.5 0.5 0.5 0.5 0.5 0.5 \
       --output_path <替换为生成图片存放的文件夹> \
       --weight_path <替换为你的预训练模型路径> \
       --model_type ffhq-config-f \
       --size 1024 \
       --style_dim 512 \
       --n_mlp 8 \
       --channel_multiplier 2 \
       --cpu
```

**参数说明:**
- latent1: 第一个风格向量的路径。可来自于Pixel2Style2Pixel生成的`dst.npy`或StyleGANv2 Fitting模块生成的`dst.fitting.npy`
- latent2: 第二个风格向量的路径。来源同第一个风格向量
- weights: 两个风格向量在不同的层次按不同比例进行混合。对于1024的分辨率，有18个层次，512的分辨率，有16个层次，以此类推。越前面，越影响混合图像的整体。越后面，越影响混合图像的细节。
           在下图中我们展示了不同权重的融合结果，可供参考。
- output_path: 生成图片存放的文件夹
- weight_path: 预训练模型路径
- model_type: PaddleGAN内置模型类型，若输入PaddleGAN已存在的模型类型，`weight_path`将失效。当前建议使用: `ffhq-config-f`
- size: 模型参数，输出图片的分辨率
- style_dim: 模型参数，风格z的维度
- n_mlp: 模型参数，风格z所输入的多层感知层的层数
- channel_multiplier: 模型参数，通道乘积，影响模型大小和生成图片质量
- cpu: 是否使用cpu推理，若不使用，请在命令中去除

## 混合结果展示

第一个风格向量对应的图像:

<div align="center">
    <img src="../../imgs/stylegan2fitting-sample.png" width="300"/>
</div>

第二个风格向量对应的图像:

<div align="center">
    <img src="../../imgs/stylegan2fitting-sample2.png" width="256"/>
</div>

两个风格向量按特定比例混合的结果:

<div align="center">
    <img src="../../imgs/stylegan2mixing-sample.png" width="256"/>
</div>

## 不同权重拟合结果展示
第一个风格向量对应的图像:

<div align="center">
    <img src="https://user-images.githubusercontent.com/50691816/130604304-292e2de4-5dc3-4613-a355-ff6163f9390f.png" width="300"/>
</div>

第二个风格向量对应的图像:

<div align="center">
    <img src="https://user-images.githubusercontent.com/50691816/130604334-3550d429-742a-4b12-a445-e54c867dbd24.png" width="256"/>
</div>

不同权重的混合结果:
<div align="center">
    <img src="https://user-images.githubusercontent.com/50691816/130603897-05f76968-bfdd-4bca-a00c-417a6e1d70af.png" height="256"/>
</div>

# 参考文献

- 1. [Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958)

  ```
  @article{Karras2019stylegan2,
    title={Analyzing and Improving the Image Quality of {StyleGAN}},
    author={Tero Karras and Samuli Laine and Miika Aittala and Janne Hellsten and Jaakko Lehtinen and Timo Aila},
    booktitle={Proc. CVPR},
    year={2020}
  }
  ```
- 2. [Encoding in Style: a StyleGAN Encoder for Image-to-Image Translation](hhttps://arxiv.org/abs/2008.00951)

  ```
  @article{richardson2020encoding,
    title={Encoding in Style: a StyleGAN Encoder for Image-to-Image Translation},
    author={Richardson, Elad and Alaluf, Yuval and Patashnik, Or and Nitzan, Yotam and Azar, Yaniv and Shapiro, Stav and Cohen-Or, Daniel},
    journal={arXiv preprint arXiv:2008.00951},
    year={2020}
  }
  ```
