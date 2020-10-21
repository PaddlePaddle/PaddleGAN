
## Getting started with PaddleGAN

### Train
```
python -u tools/main.py --config-file configs/cyclegan_cityscapes.yaml
```

continue train from last checkpoint
```
python -u tools/main.py --config-file configs/cyclegan_cityscapes.yaml --resume your_checkpoint_path
```

multiple gpus train:
```
CUDA_VISIBLE_DEVICES=0,1 python -m paddle.distributed.launch tools/main.py --config-file configs/pix2pix_cityscapes.yaml
```

### Evaluate
```
python tools/main.py --config-file configs/cyclegan_cityscapes.yaml --evaluate-only --load your_weight_path
```
