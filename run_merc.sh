export NVIDIA_TF32_OVERRIDE=0
#python -u tools/main.py --config-file configs/msvsr_reds.yaml --amp --amp_level O1

python -m paddle.distributed.launch --gpus=0,1,2,3,4,5,6,7 tools/main.py --config-file configs/msvsr_reds.yaml #--amp --amp_level O2 --seed 123 
#--resume /root/paddlejob/workspace/work/niuliling/PaddleGAN/o1_8_output_dir/msvsr_reds-2023-01-12-19-28/iter_115000_checkpoint.pdparams 
