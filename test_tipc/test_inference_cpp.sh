#!/bin/bash
source test_tipc/common_func.sh

FILENAME=$1
MODE=$2
dataline=$(awk 'NR==1, NR==18{print}'  $FILENAME)

# parser params
IFS=$'\n'
lines=(${dataline})

# parser cpp inference params
model_name=$(func_parser_value "${lines[1]}")
infer_cmd=$(func_parser_value "${lines[2]}")
model_path=$(func_parser_value "${lines[3]}")
param_path=$(func_parser_value "${lines[4]}")
video_path=$(func_parser_value "${lines[5]}")
output_dir=$(func_parser_value "${lines[6]}")
frame_num=$(func_parser_value "${lines[7]}")
device=$(func_parser_value "${lines[8]}")
gpu_id=$(func_parser_value "${lines[9]}")
use_mkldnn=$(func_parser_value "${lines[10]}")
cpu_threads=$(func_parser_value "${lines[11]}")

# only support fp32、bs=1, trt is not supported yet.
precision="fp32"
use_trt=false
batch_size=1

LOG_PATH="./test_tipc/output/${model_name}/${MODE}"
mkdir -p ${LOG_PATH}
status_log="${LOG_PATH}/results_cpp.log"

function func_cpp_inference(){
    # set log
    if [ ${device} = "GPU" ]; then
        _save_log_path="${LOG_PATH}/cpp_infer_gpu_usetrt_${use_trt}_precision_${precision}_batchsize_${batch_size}.log"
    elif [ ${device} = "CPU" ]; then
        _save_log_path="${LOG_PATH}/cpp_infer_cpu_usemkldnn_${use_mkldnn}_threads_${cpu_threads}_precision_${precision}_batchsize_${batch_size}.log"
    fi

    # set params
    set_model_path=$(func_set_params "--model_path" "${model_path}")
    set_param_path=$(func_set_params "--param_path" "${param_path}")
    set_video_path=$(func_set_params "--video_path" "${video_path}")
    set_output_dir=$(func_set_params "--output_dir" "${output_dir}")
    set_frame_num=$(func_set_params "--frame_num" "${frame_num}")
    set_device=$(func_set_params "--device" "${device}")
    set_gpu_id=$(func_set_params "--gpu_id" "${gpu_id}")
    set_use_mkldnn=$(func_set_params "--use_mkldnn" "${use_mkldnn}")
    set_cpu_threads=$(func_set_params "--cpu_threads" "${cpu_threads}")

    # run infer
    cmd="${infer_cmd} ${set_model_path} ${set_param_path} ${set_video_path} ${set_output_dir} ${set_frame_num} ${set_device} ${set_gpu_id} ${set_use_mkldnn} ${set_cpu_threads} > ${_save_log_path} 2>&1"
    eval $cmd
    last_status=${PIPESTATUS[0]}
    status_check $last_status "${cmd}" "${status_log}" "${model_name}"
}

cd deploy/cpp_infer
if [ -d "opencv-3.4.7/opencv3/" ] && [ $(md5sum opencv-3.4.7.tar.gz | awk -F ' ' '{print $1}') = "faa2b5950f8bee3f03118e600c74746a" ];then
    echo "################### build opencv skipped ###################"
else
    echo "################### building opencv ###################"
    rm -rf opencv-3.4.7.tar.gz opencv-3.4.7/
    wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/opencv-3.4.7.tar.gz
    tar -xf opencv-3.4.7.tar.gz

    cd opencv-3.4.7/
    install_path=$(pwd)/opencv3

    rm -rf build
    mkdir build
    cd build

    cmake .. \
        -DCMAKE_INSTALL_PREFIX=${install_path} \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=OFF \
        -DWITH_IPP=OFF \
        -DBUILD_IPP_IW=OFF \
        -DWITH_LAPACK=OFF \
        -DWITH_EIGEN=OFF \
        -DCMAKE_INSTALL_LIBDIR=lib64 \
        -DWITH_ZLIB=ON \
        -DBUILD_ZLIB=ON \
        -DWITH_JPEG=ON \
        -DBUILD_JPEG=ON \
        -DWITH_PNG=ON \
        -DBUILD_PNG=ON \
        -DWITH_TIFF=ON \
        -DBUILD_TIFF=ON \
        -DWITH_FFMPEG=ON

    make -j
    make install
    cd ../../
    echo "################### building opencv finished ###################"
fi

if [ -d "paddle_inference" ]; then
    echo "################### download inference lib skipped ###################"
else
    echo "################### downloading inference lib ###################"
    wget -nc https://paddle-inference-lib.bj.bcebos.com/2.3.1/cxx_c/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.1_cudnn7.6.5_trt6.0.1.5/paddle_inference.tgz
    tar -xf paddle_inference.tgz
    echo "################### downloading inference lib finished ###################"
fi

echo "################### building PaddleGAN demo ####################"
OPENCV_DIR=$(pwd)/opencv-3.4.7/opencv3
LIB_DIR=$(pwd)/paddle_inference
CUDA_LIB_DIR=$(dirname `find /usr -name libcudart.so`)
CUDNN_LIB_DIR=$(dirname `find /usr -name libcudnn.so`)
TENSORRT_DIR=''

export LD_LIBRARY_PATH=$(dirname `find ${PWD} -name libonnxruntime.so.1.11.1`):"$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH=$(dirname `find ${PWD} -name libpaddle2onnx.so.0.9.9`):"$LD_LIBRARY_PATH"

BUILD_DIR=build
rm -rf ${BUILD_DIR}
mkdir ${BUILD_DIR}
cd ${BUILD_DIR}
cmake .. \
    -DPADDLE_LIB=${LIB_DIR} \
    -DWITH_MKL=ON \
    -DWITH_GPU=ON \
    -DWITH_STATIC_LIB=OFF \
    -DWITH_TENSORRT=OFF \
    -DOPENCV_DIR=${OPENCV_DIR} \
    -DCUDNN_LIB=${CUDNN_LIB_DIR} \
    -DCUDA_LIB=${CUDA_LIB_DIR} \
    -DTENSORRT_DIR=${TENSORRT_DIR}

make -j
cd ../
echo "################### building PaddleGAN demo finished ###################"

echo "################### running test ###################"
cd ../../
func_cpp_inference
