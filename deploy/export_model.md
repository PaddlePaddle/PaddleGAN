# PaddleGAN模型导出教程

## 一、模型导出
本章节介绍如何使用`tools/export_model.py`脚本导出模型。

### 1、启动参数说明

|      FLAG      |      用途      |    默认值    |                 备注                      |
|:--------------:|:--------------:|:------------:|:-----------------------------------------:|
|       -c       |  指定配置文件  |     None     |                                           |
|       --load   |  指定加载的模型参数路径  |     None     |                                           |
|  -s|--inputs_size   |  指定模型输入形状  |     None     |                                           |
|  --output_dir  |  模型保存路径  |  `./inference_model`  |               |

### 2、使用示例

使用训练得到的模型进行试用，这里使用CycleGAN模型为例，脚本如下

```bash
# 下载预训练好的CycleGAN_horse2zebra模型
wget https://paddlegan.bj.bcebos.com/models/CycleGAN_horse2zebra.pdparams

# 导出Cylclegan模型
python -u tools/export_model.py -c configs/cyclegan_horse2zebra.yaml --load CycleGAN_horse2zebra.pdparams --inputs_size="-1,3,-1,-1;-1,3,-1,-1"
```

### 3、config配置说明
```python
export_model:
  - {name: 'netG_A', inputs_num: 1}
  - {name: 'netG_B', inputs_num: 1}
```
以上为```configs/cyclegan_horse2zebra.yaml```中的配置， 由于```CycleGAN_horse2zebra.pdparams```是个字典，需要制定其中用于导出模型的权重键值。```inputs_num```
为该网络的输入个数。

预测模型会导出到`inference_model/`目录下，分别为`cycleganmodel_netG_A.pdiparams`, `cycleganmodel_netG_A.pdiparams.info`,  `cycleganmodel_netG_A.pdmodel`, `cycleganmodel_netG_B.pdiparams`, `cycleganmodel_netG_B.pdiparams.info`, `cycleganmodel_netG_B.pdmodel`,。
