export GLOG_v=6
export FLAGS_enable_eager_mode=1
#unset GLOG_v
#python -m paddle.distributed.launch --gpus=0,1,2,3 tools/main.py --config-file configs/msvsr_reds.yaml --amp --amp_level O1
python -u tools/main.py --config-file configs/msvsr_reds.yaml --amp --amp_level O2


#python -m paddle.distributed.launch --gpus=0,1,2,3 tools/main.py --config-file configs/msvsr_reds.yaml --amp --amp_level O1
#python -u tools/main.py --config-file configs/msvsr_reds.yaml --amp --amp_level O1

