# 服务端预测部署

`PaddleGAN`训练出来的模型可以使用[Serving](https://github.com/PaddlePaddle/Serving) 部署在服务端。  
本教程以在REDS数据集上用`configs/msvsr_reds.yaml`算法训练的模型进行部署。  
预训练模型权重文件为[PP-MSVSR_reds_x4.pdparams](https://paddlegan.bj.bcebos.com/models/PP-MSVSR_reds_x4.pdparams) 。

## 1. 安装 paddle serving 
请参考[PaddleServing](https://github.com/PaddlePaddle/Serving/tree/v0.6.0) 中安装教程安装（版本>=0.6.0）。

## 2. 导出模型
PaddleGAN在训练过程包括网络的前向和优化器相关参数，而在部署过程中，我们只需要前向参数，具体参考:[导出模型](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/deploy/EXPORT_MODEL.md)

```
python tools/export_model.py -c configs/msvsr_reds.yaml   --inputs_size="1,2,3,180,320" --load /path/to/model --export_serving_model True
----output_dir /path/to/output
```

以上命令会在`/path/to/output`文件夹下生成一个`msvsr`文件夹：
```
output
│   ├── multistagevsrmodel_generator
│   │   ├── multistagevsrmodel_generator.pdiparams
│   │   ├── multistagevsrmodel_generator.pdiparams.info
│   │   ├── multistagevsrmodel_generator.pdmodel
│   │   ├── serving_client
│   │   │   ├── serving_client_conf.prototxt
│   │   │   ├── serving_client_conf.stream.prototxt
│   │   ├── serving_server
│   │   │   ├── __model__
│   │   │   ├── __params__
│   │   │   ├── serving_server_conf.prototxt
│   │   │   ├── serving_server_conf.stream.prototxt
│   │   │   ├── ...
```

`serving_client`文件夹下`serving_client_conf.prototxt`详细说明了模型输入输出信息
`serving_client_conf.prototxt`文件内容为：
```
feed_var {
  name: "lqs"
  alias_name: "lqs"
  is_lod_tensor: false
  feed_type: 1
  shape: 1
  shape: 2
  shape: 3
  shape: 180
  shape: 320
}
fetch_var {
  name: "stack_18.tmp_0"
  alias_name: "stack_18.tmp_0"
  is_lod_tensor: false
  fetch_type: 1
  shape: 1
  shape: 2
  shape: 3
  shape: 720
  shape: 1280
}
fetch_var {
  name: "stack_19.tmp_0"
  alias_name: "stack_19.tmp_0"
  is_lod_tensor: false
  fetch_type: 1
  shape: 1
  shape: 3
  shape: 720
  shape: 1280
}
```

## 4. 启动PaddleServing服务

```
cd output_dir/multistagevsrmodel_generator/

# GPU
python -m paddle_serving_server.serve --model serving_server --port 9393 --gpu_ids 0

# CPU
python -m paddle_serving_server.serve --model serving_server --port 9393
```

## 5. 测试部署的服务
```
# 进入到导出模型文件夹
cd output/msvsr/
```

设置`prototxt`文件路径为`serving_client/serving_client_conf.prototxt` 。  
设置`fetch`为`fetch=["stack_19.tmp_0"])`

测试
```
# 进入目录
cd output/msvsr/

# 测试代码 test_client.py 会自动创建output文件夹，并在output下生成`res.mp4`文件
python ../../deploy/serving/test_client.py input_video frame_num
```
