#export GLOG_v=6
python -m paddle.distributed.launch --gpus=0,1,2,3 tools/main.py --config-file configs/msvsr_reds.yaml --amp --amp_level O1

#python -u tools/main.py --config-file configs/msvsr_reds.yaml --amp --amp_level O1

