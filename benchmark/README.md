# PaddGAN模型性能复现
## 目录

```
├── README.md               # 说明文档
├── configs
│   ├── stylegan_v2_256_ffhq.yaml
│   ├── firstorder_vox_256.yaml
│   ├── esrgan_psnr_x4_div2k.yaml
│   ├── edvr_m_wo_tsa.yaml
│   ├── basicvsr_reds.yaml
├── prepare.sh    # 相关数据准备脚本，完成数据、模型的自动下载
├── benchmark.yaml          # 配置文件，设置测试模型及模型参数
├── run_all.sh              # 执行入口，测试并获取所有生成对抗模型的训练性能
└── run_benchmark.sh        # 执行实体，测试单个分割模型的训练性能  
```

#### 备注
BasicVSR模型因竞品torch模型只能测4卡，故这里也测4卡。

因REDS数据集较大，避免每次下载时间较长，需要在Docker建立好后，将REDS数据集放到/workspace/data/目录一下。

# Docker 运行环境
docker image: paddlepaddle/paddle:latest-gpu-cuda10.1-cudnn7
paddle = 2.1.2
python = 3.7

# 运行benchmark测试
git clone https://github.com/PaddlePaddle/PaddleGAN.git
cd PaddleGAN
bash benchmark/prepare.sh StyleGANv2

# 运行指定模型
Usage：bash run_benchmark.sh ${model_item} {batch_size} ${fp_item} {run_mode} ${device_num}
model_item: StyleGANv2, FOMM, esrgan, edvr, basicvsr

# 单卡
CUDA_VISIBLE_DEVICES=0 bash benchmark/run_benchmark.sh StyleGANv2 8 fp32 SingleP DP N1C1

# 多卡
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash benchmark/run_benchmark.sh StyleGANv2 8 fp32 MultiP DP N1C8

# 打开Profiling开关运行，只用跑单卡
export PROFILING=true
CUDA_VISIBLE_DEVICES=0 bash benchmark/run_benchmark.sh StyleGANv2 8 fp32 SingleP DP N1C1

# profiling 目前只跑这一种默认的配置
add_options="--profiler_options=\"batch_range=[10,20];state=GPU;tracer_option=Default;profile_path=model.profile\""  
