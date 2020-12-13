
## Getting started with PaddleGAN

Note:
* Before starting to use PaddleGAN, please make sure you have read the [install document](./install_en.md), and prepare the dataset according to the [data preparation document](./data_prepare_en.md)
* The following tutorial uses the train and evaluate of the CycleGAN model on the Cityscapes dataset as an example

### Train

#### Train with single gpu
```
python -u tools/main.py --config-file configs/cyclegan_cityscapes.yaml
```
#### Args

- `--config-file (str)`: path of config file。

The output log, weight, and visualization result will be saved in ```./output_dir``` by default, which can be modified by the ```output_dir``` parameter in the config file:
 ```
 output_dir: output_dir
 ```



The saved folder will automatically generate a new directory based on the model name and timestamp. The directory example is as follows:
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

Also, you can add the parameter ```enable_visualdl: true``` in the configuration file, use [PaddlePaddle VisualDL](https://github.com/PaddlePaddle/VisualDL) record the metrics or images generated in the training process, and run the command to monitor the training process:
```
visualdl --logdir output_dir/CycleGANModel-2020-10-29-09-21/
```

#### Recovery of training

The checkpoint of the previous epoch will be saved by default during the training process to facilitate the recovery of training
```
python -u tools/main.py --config-file configs/cyclegan_cityscapes.yaml --resume your_checkpoint_path
```
#### Args

- `--resume (str)`: path of checkpoint。

#### Train with multiple gpus:
```
CUDA_VISIBLE_DEVICES=0,1 python -m paddle.distributed.launch tools/main.py --config-file configs/cyclegan_cityscapes.yaml
```

### evaluate
```
python tools/main.py --config-file configs/cyclegan_cityscapes.yaml --evaluate-only --load your_weight_path
```

#### Args
- `--evaluate-only`: whether to evaluate only。
- `--load (str)`: path of weight。
