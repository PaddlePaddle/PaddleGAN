# source /ssd1/vis/liufanglong/.bashrc
#export PATH=/home/work/cuda-9.0/bin:$PATH
#export PATH=/home/work/cuda-9.0/bin:$PATH
#export LD_LIBRARY_PATH="/home/work/cuda-9.0/lib64:$LD_LIBRARY_PATH"
#export LD_LIBRARY_PATH=/home/vis/chao/local/cudnn_v7.6/cuda/lib64:$LD_LIBRARY_PATH
#export CPLUS_INCLUDE_PATH=/home/vis/chao/local/cudnn_v7.6/cuda/include:/ssd1/vis/liufanglong/local/fluid_1.1.0_for_slurm/nccl_2.3.5/include:$CPLUS_INCLUDE_PATH
#export LD_LIBRARY_PATH=/ssd1/vis/liufanglong/local/fluid_1.1.0_for_slurm/nccl_2.3.5/lib:$LD_LIBRARY_PATH

include_dir=$( python -c 'import paddle; print(paddle.sysconfig.get_include())' )
lib_dir=$( python -c 'import paddle; print(paddle.sysconfig.get_lib())' )

echo $include_dir
echo $lib_dir

OPS='correlation_op'
for op in ${OPS}
do
nvcc ${op}.cu -c -o ${op}.cu.o -ccbin cc -DPADDLE_WITH_CUDA -DEIGEN_USE_GPU -DPADDLE_USE_DSO -DPADDLE_WITH_MKLDNN -Xcompiler -fPIC -std=c++11 -Xcompiler -fPIC -w --expt-relaxed-constexpr -O0 -g -DNVCC \
    -I ${include_dir}/third_party/ \
    -I ${include_dir}
done

# g++-4.8 correlation_op.cu.o correlation_op.cc -o correlation_lib.so -DPADDLE_WITH_MKLDNN -shared -fPIC -std=c++11 -O0 -g \
# g++ ${OPS}.cu.o ${OPS}.cc -o correlation_lib.so -DPADDLE_WITH_MKLDNN -shared -fPIC -std=c++11 -O0 -g \
g++ correlation_op.cu.o correlation_op.cc -o correlation_lib.so -DPADDLE_WITH_MKLDNN -shared -fPIC -std=c++11 -O0 -g \
  -I ${include_dir}/third_party/ \
  -I ${include_dir} \
  -L ${lib_dir} \
  -L /usr/local/cuda/lib64/ -lpaddle_framework -lcudart

# rm *.cu.o
