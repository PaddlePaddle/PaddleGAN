
# prepare
pip install argparse
pip install -v -e .
# Download test dataset and save it to PaddleGAN/data
# It automatic downloads the pretrained models saved in ~/.paddlegan
data_item=${1:-'StyleGANv2'} 
case ${data_item} in
    StyleGANv2)    # 1 数据集<500M ,可直接打包放在各个套件自己的bce上。
        wget https://paddlegan.bj.bcebos.com/datasets/ffhq.tar --no-check-certificate \
        -O data/ffhq.tar
        tar -vxf data/ffhq.tar -C data/  ;;
    FOMM)
        wget https://paddlegan.bj.bcebos.com/datasets/fom_test_data.tar  --no-check-certificate \
        -O data/fom_test_data.tar
        tar -vxf data/fom_test_data.tar -C data/  ;;
    esrgan)
        wget https://paddlegan.bj.bcebos.com/datasets/DIV2KandSet14.tar  --no-check-certificate \
        -O data/DIV2KandSet14.tar
        tar -vxf data/DIV2KandSet14.tar -C data/  ;;
    edvr|basicvsr)
        mkdir -p data/REDS 
        python ${BENCHMARK_ROOT}/paddlecloud/file_upload_download.py \
        --remote-path frame_benchmark/paddle/PaddleGAN/REDS/test_sharp \
        --local-path ./data/REDS \
        --mode download
        python ${BENCHMARK_ROOT}/paddlecloud/file_upload_download.py \
        --remote-path frame_benchmark/paddle/PaddleGAN/REDS/test_sharp_bicubic \
        --local-path ./data/REDS \
        --mode download
        python ${BENCHMARK_ROOT}/paddlecloud/file_upload_download.py \
        --remote-path frame_benchmark/paddle/PaddleGAN/REDS/train_sharp \
        --local-path ./data/REDS \
        --mode download
        python ${BENCHMARK_ROOT}/paddlecloud/file_upload_download.py \
        --remote-path frame_benchmark/paddle/PaddleGAN/REDS/train_sharp_bicubic \
        --local-path ./data/REDS \
        --mode download
        tar -vxf data/REDS/train_sharp.tar -C data/REDS
        tar -vxf data/REDS/train_sharp_bicubic.tar -C data/REDS
        tar -vxf data/REDS/REDS4_test_sharp.tar -C data/REDS
        tar -vxf data/REDS/REDS4_test_sharp_bicubic.tar -C data/REDS
        wget https://paddlegan.bj.bcebos.com/datasets/meta_info_REDS_GT.tar  --no-check-certificate \
        -O data/REDS/meta_info_REDS_GT.tar
        tar -vxf data/REDS/meta_info_REDS_GT.tar -C data/REDS
        echo "download data" #waiting data process
        echo "dataset prepared done"  ;;
    *) echo "choose data_item"; exit 1;
esac