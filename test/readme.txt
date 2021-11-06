## 本路径下所有文件均为 photopen 模型的训练过程、预测过程的测试使用。

# 使用预训练模型预测
python applications/tools/photopen.py --semantic_label_path test/sem.png  --weight_path test/generator.pdparams --output_path output_dir/pic.jpg --config-file configs/photopen.yaml

# 使用checkpoint预测
python applications/tools/photopen.py --semantic_label_path test/sem.png  --weight_path output_dir/photopen-2021-10-05-14-38/iter_1_weight.pdparams --output_path output_dir/pic.jpg --config-file configs/photopen.yaml


# 训练
python -u tools/main.py --config-file configs/photopen.yaml

# 继续训练
python -u tools/main.py --config-file configs/photopen.yaml --resume output_dir/photopen-2021-09-30-15-59/iter_3_checkpoint.pdparams

# 训练，覆盖参数
python -u tools/main.py --config-file configs/photopen.yaml --o model.generator.ngf=1 model.discriminator.ndf=1

# 测试
python -u tools/main.py --config-file configs/photopen.yaml --evaluate-only --load output_dir/photopen-2021-11-06-20-59/iter_1_checkpoint.pdparams
