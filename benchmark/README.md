# PaddGAN模型性能复现
## 目录

```
├── README.md               # 说明文档
├── benchmark.yaml          # 配置文件，设置测试模型及模型参数
├── run_all.sh              # 执行入口，测试并获取所有生成对抗模型的训练性能
└── run_benchmark.sh        # 执行实体，测试单个分割模型的训练性能  
```

## 环境介绍
### 物理机环境
- 单机（单卡、8卡）
  - 系统：CentOS release 7.5 (Final)
  - GPU：Tesla V100-SXM2-32GB * 8
  - CPU：Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz * 80
  - CUDA、cudnn Version: cuda10.2-cudnn7

#### 备注
BasicVSR模型因竞品torch模型只能测4卡，故这里也测4卡。

因REDS数据集较大，避免每次下载时间较长，需要在Docker建立好后，将REDS数据集放到/workspace/data/目录一下。

### Docker 镜像

- **镜像版本**: `registry.baidubce.com/paddlepaddle/paddle:2.1.2-gpu-cuda10.2-cudnn7`
- **paddle 版本**: `2.1.2`
- **CUDA 版本**: `10.2`
- **cuDnn 版本**: `7`

## 在PaddleGAN目录下，启动测试脚本的方法如下：
```script
ImageName="registry.baidubce.com/paddlepaddle/paddle:2.1.2-gpu-cuda10.2-cudnn7"
docker pull ${ImageName}

run_cmd="set -xe;
        cd /workspace ;
        bash -x benchmark/run_all.sh"

nvidia-docker run --name test_paddlegan -i  \
    --net=host \
    --shm-size=128g \
    -v $PWD:/workspace \
    ${ImageName}  /bin/bash -c "${run_cmd}"
```

如果需要打开profile选项，可以直接替换`run_cmd`为：
```
run_cmd="set -xe;
        cd /workspace ;
        bash -x benchmark/run_all.sh on"
```

## 输出

执行完成后，在PaddleGAN目录会产出模型训练性能数据的文件，比如`esrgan_mp_bs32_fp32_8`等文件。
