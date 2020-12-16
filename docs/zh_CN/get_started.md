
## 快速开始使用PaddleGAN

注意：
* 开始使用PaddleGAN前请确保已经阅读过[安装文档](./install.md)，并根据[数据准备文档](./data_prepare.md)准备好数据集。
* 以下教程以CycleGAN模型在Cityscapes数据集上的训练预测作为示例。


### 训练

#### 单卡训练
```
python -u tools/main.py --config-file configs/cyclegan_cityscapes.yaml
```
#### 参数

- `--config-file (str)`: 配置文件的路径。

 输出的日志，权重，可视化结果会默认保存在```./output_dir```中，可以通过配置文件中的```output_dir```参数修改：
 ```
 output_dir: output_dir
 ```

 保存的文件夹会根据模型名字和时间戳自动生成一个新目录，目录示例如下：
```
output_dir
└── CycleGANModel-2020-10-29-09-21
    ├── epoch_1_checkpoint.pkl
    ├── log.txt
    └── visual_train
        ├── epoch001_fake_A.png
        ├── epoch001_fake_B.png
        ├── epoch001_idt_A.png
        ├── epoch001_idt_B.png
        ├── epoch001_real_A.png
        ├── epoch001_real_B.png
        ├── epoch001_rec_A.png
        ├── epoch001_rec_B.png
        ├── epoch002_fake_A.png
        ├── epoch002_fake_B.png
        ├── epoch002_idt_A.png
        ├── epoch002_idt_B.png
        ├── epoch002_real_A.png
        ├── epoch002_real_B.png
        ├── epoch002_rec_A.png
        └── epoch002_rec_B.png
```
同时可以通过在配置文件中添加参数```enable_visualdl: true```使用[飞桨VisualDL](https://github.com/PaddlePaddle/VisualDL)对训练过程产生的指标或生成的图像进行记录，并运行相应命令对训练过程进行实时监控：
```
visualdl --logdir output_dir/CycleGANModel-2020-10-29-09-21/
```

#### 恢复训练

训练过程中默认会保存上一个epoch的checkpoint，方便恢复训练
```
python -u tools/main.py --config-file configs/cyclegan_cityscapes.yaml --resume your_checkpoint_path
```
#### 参数

- `--resume (str)`: 用来恢复训练的checkpoint路径。

#### 多卡训练:
```
CUDA_VISIBLE_DEVICES=0,1 python -m paddle.distributed.launch tools/main.py --config-file configs/cyclegan_cityscapes.yaml
```

### 预测
```
python tools/main.py --config-file configs/cyclegan_cityscapes.yaml --evaluate-only --load your_weight_path
```

#### 参数
- `--evaluate-only`: 是否仅进行预测。
- `--load (str)`: 训练好的权重路径。
