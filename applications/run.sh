cd DAIN/pwcnet/correlation_op
# 第一次需要执行 
# bash make.shap
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`python -c 'import paddle; print(paddle.sysconfig.get_lib())'`
export PYTHONPATH=$PYTHONPATH:`pwd`
cd -

# input 输入视频的路径
# output 输出视频保存的路径
# proccess_order 使用模型的顺序

python tools/main.py \
--input input.mp4  --output output --proccess_order DAIN DeOldify EDVR
