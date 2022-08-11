#!/bin/bash

# echo "################### install autolog ###################"
# python -m pip install https://paddleocr.bj.bcebos.com/libs/auto_log-1.2.0-py3-none-any.whl

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
    wget -nc https://paddle-inference-lib.bj.bcebos.com/2.2.2/cxx_c/Linux/GPU/x86-64_gcc5.4_avx_mkl_cuda10.1_cudnn7.6.5_trt6.0.1.5/paddle_inference.tgz
    tar -xf paddle_inference.tgz
    echo "################### downloading inference lib finished ###################"
fi

echo "################### building PaddleGAN demo ####################"
OPENCV_DIR=$(pwd)/opencv-3.4.7/opencv3
echo ${OPENCV_DIR}
LIB_DIR=$(pwd)/paddle_inference
echo ${LIB_DIR}
CUDA_LIB_DIR=$(dirname `find /usr -name libcudart.so`)
CUDNN_LIB_DIR=$(dirname `find /usr -name libcudnn.so`)
TENSORRT_DIR=''

BUILD_DIR=build
rm -rf ${BUILD_DIR}
mkdir ${BUILD_DIR}
cd ${BUILD_DIR}
cmake .. \
    -DPADDLE_LIB=${LIB_DIR} \
    -DWITH_MKL=ON \
    -DWITH_GPU=OFF \
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
./build/vsr \
    --model_path=/workspace/PaddleGAN/inference/msvsr/multistagevsrmodel_generator.pdmodel \
    --param_path=/workspace/PaddleGAN/inference/msvsr/multistagevsrmodel_generator.pdiparams \
    --frame_num=2 \
    --video_path=/workspace/PaddleGAN/data/low_res.mp4 \
    --output_dir=/workspace/PaddleGAN/test_tipc/output/ \
    --device="CPU" \
    --use_mkldnn=true
