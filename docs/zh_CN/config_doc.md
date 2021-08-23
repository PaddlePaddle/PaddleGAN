# 配置文件说明文档

## Config文件参数介绍

以`lapstyle_rev_first.yaml`为例。

### Global

| 字段                      | 用途                       | 默认值          |
| ------------------------- | -------------------------- | --------------- |
| total_iters               | 设置总训练步数             | 30000           |
| min_max                   | tensor数值范围（存图像时使用） | (0., 1.)        |
| output_dir                | 设置输出结果所在的文件路径 | ./output_dir    |
| snapshot_config: interval | 设置保存模型参数的间隔       | 5000            |

### Model

| 字段                    | 用途     | 默认值 |
| :---------------------- | -------- | ------ |
| name                    | 模型名称 | LapStyleRevFirstModel |
| revnet_generator        | 设置revnet生成器 | RevisionNet |
| revnet_discriminator    | 设置revnet判别器 | LapStyleDiscriminator |
| draftnet_encode         | 设置draftnet编码器 | Encoder |
| draftnet_decode         | 设置draftnet解码器 | DecoderNet |
| calc_style_emd_loss     | 设置style损失1 | CalcStyleEmdLoss |
| calc_content_relt_loss  | 设置content损失1 | CalcContentReltLoss |
| calc_content_loss       | 设置content损失2 | CalcContentLoss |
| calc_style_loss         | 设置style损失2 | CalcStyleLoss |
| gan_criterion: name     | 设置GAN损失 |  GANLoss |
| gan_criterion: gan_mode | 设置GAN损失模态参数 | vanilla |
| content_layers          | 设置计算content损失2的网络层 |['r11', 'r21', 'r31', 'r41', 'r51']|
| style_layers            | 设置计算style损失2的网络层 | ['r11', 'r21', 'r31', 'r41', 'r51'] |
| content_weight          | 设置content总损失权重 | 1.0 |
| style_weigh             | 设置style总损失权重 | 3.0 |

### Dataset (train & test)

| 字段         | 用途                 | 默认值               |
| :----------- | -------------------- | -------------------- |
| name         | 数据集名称           | LapStyleDataset      |
| content_root | 数据集所在路径       | data/coco/train2017/ |
| style_root   | 目标风格图片所在路径 | data/starrynew.png   |
| load_size    | 输入图像resize后图像大小 | 280          |
| crop_size    | 随机剪裁图像后图像大小 | 256           |
| num_workers  | 设置工作进程个数 | 16                   |
| batch_size   | 设置一次训练所抓取的数据样本数量 | 5                    |

### Lr_scheduler 

| 字段          | 用途             | 默认值         |
| :------------ | ---------------- | -------------- |
| name          | 学习策略名称 | NonLinearDecay |
| learning_rate | 设置初始学习率   | 1e-4           |
| lr_decay      | 设置学习率衰减率 | 5e-5           |

### Optimizer

| 字段      | 用途       | 默认值  |
| :-------- | ---------- | ------- |
| name      | 优化器类名 | Adam    |
| net_names | 优化器作用的网络 | net_rev |
| beta1     | 设置优化器参数beta1  | 0.9     |
| beta2     | 设置优化器参数beta2 | 0.999   |

### Validate

| 字段     | 用途 | 默认值 |
| :------- | ---- | ------ |
| interval | 设置验证间隔 | 500    |
| save_img | 验证时是否保存图像 | false  |

### Log_config

| 字段             | 用途 | 默认值 |
| :--------------- | ---- | ------ |
| interval         | 设置打印log间隔 | 10     |
| visiual_interval | 设置训练过程中保存生成图像的间隔 | 500    |
