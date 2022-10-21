#!/bin/bash
FILENAME=$1

# MODE be one of ['lite_train_lite_infer' 'lite_train_whole_infer' 'whole_train_whole_infer',
#                 'whole_infer', 'benchmark_train', 'cpp_infer']

MODE=$2

dataline=$(cat ${FILENAME})

# parser params
IFS=$'\n'
lines=(${dataline})
function func_parser_key(){
    strs=$1
    IFS=":"
    array=(${strs})
    tmp=${array[0]}
    echo ${tmp}
}
function func_parser_value(){
    strs=$1
    IFS=":"
    array=(${strs})
    tmp=${array[1]}
    echo ${tmp}
}
IFS=$'\n'
# The training params
model_name=$(func_parser_value "${lines[1]}")
trainer_list=$(func_parser_value "${lines[14]}")

if [ ${MODE} = "benchmark_train" ];then
    pip install -r requirements.txt
    MODE="lite_train_lite_infer"
fi

if [ ${MODE} = "lite_train_lite_infer" ];then

    case ${model_name} in
    Pix2pix)
        rm -rf ./data/pix2pix*
        wget -nc -P ./data/ https://paddlegan.bj.bcebos.com/datasets/pix2pix_facade_lite.tar --no-check-certificate
        cd ./data/ && tar xf pix2pix_facade_lite.tar && cd ../ ;;
    CycleGAN)
        rm -rf ./data/cyclegan*
        wget -nc -P ./data/ https://paddlegan.bj.bcebos.com/datasets/cyclegan_horse2zebra_lite.tar --no-check-certificate
        cd ./data/ && tar xf cyclegan_horse2zebra_lite.tar && cd ../ ;;
    StyleGANv2)
        rm -rf ./data/ffhq*
        wget -nc -P ./data/ https://paddlegan.bj.bcebos.com/datasets/ffhq_256.tar --no-check-certificate
        cd ./data/ && tar xf ffhq_256.tar && cd ../ ;;
    FOMM)
        rm -rf ./data/fom_lite*
        wget -nc -P ./data/ https://paddlegan.bj.bcebos.com/datasets/fom_lite.tar  --no-check-certificate --no-check-certificate
        cd ./data/ && tar xf fom_lite.tar && cd ../ ;;
    edvr|basicvsr|msvsr)
        rm -rf ./data/reds*
        wget -nc -P ./data/ https://paddlegan.bj.bcebos.com/datasets/reds_lite.tar --no-check-certificate
        cd ./data/ && tar xf reds_lite.tar && cd ../ ;;
    esrgan)
        rm -rf ./data/DIV2K*
        wget -nc -P ./data/ https://paddlegan.bj.bcebos.com/datasets/DIV2KandSet14paddle.tar --no-check-certificate
        cd ./data/ && tar xf DIV2KandSet14paddle.tar && cd ../ ;;
    swinir)
        rm -rf ./data/*sets
        wget -nc -P ./data/ https://paddlegan.bj.bcebos.com/datasets/swinir_data.zip --no-check-certificate
        cd ./data/ && unzip -q swinir_data.zip && cd ../ ;;
    invdn)
        rm -rf ./data/SIDD_*
        wget -nc -P ./data/ https://paddlegan.bj.bcebos.com/datasets/SIDD_mini.zip --no-check-certificate
        cd ./data/ && unzip -q SIDD_mini.zip && cd ../ ;;
    nafnet)
        rm -rf ./data/SIDD*
        wget -nc -P ./data/ https://paddlegan.bj.bcebos.com/datasets/SIDD_mini.zip --no-check-certificate
        cd ./data/ && unzip -q SIDD_mini.zip && mkdir -p SIDD && mv ./SIDD_Medium_Srgb_Patches_512/* ./SIDD/ \
        && mv ./SIDD_Valid_Srgb_Patches_256/* ./SIDD/ && mv ./SIDD/valid ./SIDD/val \
        && mv ./SIDD/train/GT ./SIDD/train/target && mv ./SIDD/train/Noisy ./SIDD/train/input \
        && mv ./SIDD/val/Noisy ./SIDD/val/input && mv ./SIDD/val/GT ./SIDD/val/target && cd ../ ;;
    singan)
        rm -rf ./data/singan*
        wget -nc -P ./data/ https://paddlegan.bj.bcebos.com/datasets/singan-official_images.zip --no-check-certificate
        cd ./data/ && unzip -q singan-official_images.zip && cd ../
        mkdir -p ./data/singan
        mv ./data/SinGAN-official_images/Images/stone.png ./data/singan ;;
    GFPGAN)
        rm -rf ./data/gfpgan*
        wget -nc -P ./data/ https://paddlegan.bj.bcebos.com/datasets/gfpgan_tipc_data.zip --no-check-certificate
        mkdir -p ./data/gfpgan_data
        cd ./data/ && unzip -q gfpgan_tipc_data.zip -d gfpgan_data/ && cd ../ ;;
    aotgan)
        rm -rf ./data/aotgan*
        wget -nc -P ./data/ https://paddlegan.bj.bcebos.com/datasets/aotgan.zip --no-check-certificate
        cd ./data/ && unzip -q aotgan.zip && cd ../ ;;
    esac
elif [ ${MODE} = "whole_train_whole_infer" ];then
    if [ ${model_name} == "Pix2pix" ]; then
        rm -rf ./data/facades*
        wget -nc -P ./data/ http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz --no-check-certificate
        cd ./data/ && tar -xzf facades.tar.gz && cd ../
    elif [ ${model_name} == "CycleGAN" ]; then
        rm -rf ./data/horse2zebra*
        wget -nc -P ./data/ https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip --no-check-certificate
        cd ./data/ && unzip horse2zebra.zip && cd ../
    elif [ ${model_name} == "singan" ]; then
        rm -rf ./data/singan*
        wget -nc -P ./data/ https://paddlegan.bj.bcebos.com/datasets/singan-official_images.zip --no-check-certificate
        cd ./data/ && unzip -q singan-official_images.zip && cd ../
        mkdir -p ./data/singan
        mv ./data/SinGAN-official_images/Images/stone.png ./data/singan
    fi
elif [ ${MODE} = "lite_train_whole_infer" ];then
    if [ ${model_name} == "Pix2pix" ]; then
        rm -rf ./data/facades*
        wget -nc -P ./data/ https://paddlegan.bj.bcebos.com/datasets/pix2pix_facade_lite.tar --no-check-certificate
        cd ./data/ && tar xf pix2pix_facade_lite.tar && cd ../
    elif [ ${model_name} == "CycleGAN" ]; then
        rm -rf ./data/horse2zebra*
        wget -nc -P ./data/ https://paddlegan.bj.bcebos.com/datasets/cyclegan_horse2zebra_lite.tar  --no-check-certificate --no-check-certificate
        cd ./data/ && tar xf cyclegan_horse2zebra_lite.tar && cd ../
    elif [ ${model_name} == "FOMM" ]; then
        rm -rf ./data/first_order*
        wget -nc -P ./data/ https://paddlegan.bj.bcebos.com/datasets/fom_lite.tar  --no-check-certificate --no-check-certificate
        cd ./data/ && tar xf fom_lite.tar && cd ../
    elif [ ${model_name} == "StyleGANv2" ]; then
        rm -rf ./data/ffhq*
        wget -nc -P ./data/ https://paddlegan.bj.bcebos.com/datasets/ffhq_256.tar --no-check-certificate
        cd ./data/ && tar xf ffhq_256.tar && cd ../
    elif [ ${model_name} == "basicvsr" ]; then
        rm -rf ./data/reds*
        wget -nc -P ./data/ https://paddlegan.bj.bcebos.com/datasets/reds_lite.tar --no-check-certificate
        cd ./data/ && tar xf reds_lite.tar && cd ../
    elif [ ${model_name} == "msvsr" ]; then
        rm -rf ./data/reds*
        wget -nc -P ./data/ https://paddlegan.bj.bcebos.com/datasets/reds_lite.tar --no-check-certificate
        cd ./data/ && tar xf reds_lite.tar && cd ../
    elif [ ${model_name} == "singan" ]; then
        rm -rf ./data/singan*
        wget -nc -P ./data/ https://paddlegan.bj.bcebos.com/datasets/singan-official_images.zip --no-check-certificate
        cd ./data/ && unzip -q singan-official_images.zip && cd ../
        mkdir -p ./data/singan
        mv ./data/SinGAN-official_images/Images/stone.png ./data/singan
    fi
elif [ ${MODE} = "whole_infer" ];then
    if [ ${model_name} = "Pix2pix" ]; then
        rm -rf ./data/facades*
        rm -rf ./inference/pix2pix*
        wget -nc  -P ./inference https://paddlegan.bj.bcebos.com/static_model/pix2pix_facade.tar --no-check-certificate
        wget -nc  -P ./data https://paddlegan.bj.bcebos.com/datasets/facades_test.tar --no-check-certificate
        cd ./data && tar xf facades_test.tar && mv facades_test facades && cd ../
        cd ./inference && tar xf pix2pix_facade.tar && cd ../
    elif [ ${model_name} = "CycleGAN" ]; then
        rm -rf ./data/cyclegan*
        rm -rf ./inference/cyclegan*
        wget -nc  -P ./inference https://paddlegan.bj.bcebos.com/static_model/cyclegan_horse2zebra.tar  --no-check-certificate
        wget -nc  -P ./data https://paddlegan.bj.bcebos.com/datasets/cyclegan_horse2zebra_test.tar  --no-check-certificate
        cd ./data && tar xf cyclegan_horse2zebra_test.tar && mv cyclegan_test horse2zebra && cd ../
        cd ./inference && tar xf cyclegan_horse2zebra.tar && cd ../
    elif [ ${model_name} == "FOMM" ]; then
        rm -rf ./data/first_order*
        rm -rf ./inference/fom_dy2st*
        wget -nc -P ./data/ https://paddlegan.bj.bcebos.com/datasets/fom_lite_test.tar  --no-check-certificate
        wget -nc  -P ./inference https://paddlegan.bj.bcebos.com/static_model/fom_dy2st.tar --no-check-certificate
        cd ./data/ && tar xf fom_lite_test.tar && cd ../
        cd ./inference && tar xf fom_dy2st.tar && cd ../
    elif [ ${model_name} == "StyleGANv2" ]; then
        rm -rf ./data/ffhq*
        wget -nc -P ./data/ https://paddlegan.bj.bcebos.com/datasets/ffhq_256.tar --no-check-certificate
        wget -nc  -P ./inference https://paddlegan.bj.bcebos.com/static_model/stylegan2_1024.tar --no-check-certificate
        cd ./inference && tar xf stylegan2_1024.tar && cd ../
        cd ./data/ && tar xf ffhq_256.tar && cd ../
    elif [ ${model_name} == "basicvsr" ]; then
        rm -rf ./data/reds*
        rm -rf ./inference/basic*
        wget -nc -P ./data/ https://paddlegan.bj.bcebos.com/datasets/reds_lite.tar --no-check-certificate
        wget -nc  -P ./inference https://paddlegan.bj.bcebos.com/static_model/basicvsr.tar --no-check-certificate
        cd ./inference && tar xf basicvsr.tar && cd ../
        cd ./data/ && tar xf reds_lite.tar && cd ../
    elif [ ${model_name} == "msvsr" ]; then
        rm -rf ./data/reds*
        rm -rf ./inference/msvsr*
        wget -nc -P ./data/ https://paddlegan.bj.bcebos.com/datasets/reds_lite.tar --no-check-certificate
        wget -nc  -P ./inference https://paddlegan.bj.bcebos.com/static_model/msvsr.tar --no-check-certificate
        cd ./inference && tar xf msvsr.tar && cd ../
        cd ./data/ && tar xf reds_lite.tar && cd ../
    elif [ ${model_name} == "singan" ]; then
        rm -rf ./data/singan*
        wget -nc -P ./data/ https://paddlegan.bj.bcebos.com/datasets/singan-official_images.zip --no-check-certificate
        wget -nc -P ./inference https://paddlegan.bj.bcebos.com/datasets/singan.zip --no-check-certificate
        cd ./data/ && unzip -q singan-official_images.zip && cd ../
        cd ./inference/ && unzip -q singan.zip && cd ../
        mkdir -p ./data/singan
        mv ./data/SinGAN-official_images/Images/stone.png ./data/singan
    fi
elif [ ${MODE} = "cpp_infer" ]; then
    if [ ${model_name} == "msvsr" ]; then
        rm -rf ./inference/msvsr*
        wget -nc  -P ./inference https://paddlegan.bj.bcebos.com/static_model/msvsr.tar --no-check-certificate
        cd ./inference && tar xf msvsr.tar && cd ../
        wget -nc -P ./data https://paddlegan.bj.bcebos.com/datasets/low_res.mp4 --no-check-certificate
    fi
fi
