## 本路径下所有文件均为 aotgan 模型的训练过程、预测过程的测试使用。

# 使用预训练模型预测(预训练权重 g.pdparams 从 https://aistudio.baidu.com/aistudio/datasetdetail/165081 下载)
python applications/tools/aotgan.py --input_image_path test/aotgan/armani1.jpg --input_mask_path test/aotgan/armani1.png  --weight_path test/aotgan/g.pdparams --output_path output_dir/armani_pred.jpg --config-file configs/aotgan.yaml

# 使用checkpoint预测
python applications/tools/aotgan.py --input_image_path test/aotgan/armani1.jpg --weight_path output_dir/aotgan-2022-10-08-18-00/iter_200_weight.pdparams --input_mask_path test/aotgan/armani1.png --output_path output_dir/armani_pred.jpg --config-file configs/aotgan.yaml

# 训练
python -u tools/main.py --config-file configs/aotgan.yaml

# 继续训练
python -u tools/main.py --config-file configs/aotgan.yaml --resume  output_dir/aotgan-2022-10-08-18-00/iter_200_checkpoint.pdparams

# 训练，覆盖参数
python -u tools/main.py --config-file configs/aotgan.yaml --o dataset.train.batch_size=2 model.img_size=256 dataset.train.img_size=256 total_iters=1000 snapshot_config.interval=100

# 测试
python -u tools/main.py --config-file configs/aotgan.yaml --evaluate-only --load output_dir/aotgan-2022-10-08-18-00/iter_200_checkpoint.pdparams
