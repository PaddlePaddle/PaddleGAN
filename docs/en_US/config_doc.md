# Instruction of Config Files

## Introduction of Parameters

Take`lapstyle_rev_first.yaml` as an example.

### Global

| Field                 | Usage            | Default |
| ------------------------- | :------------------------- | --------------- |
| total_iters               | total training steps                         | 30000           |
| min_max                   | numeric range of tensor（for image storage） | (0., 1.)        |
| output_dir                | path of the output                           | ./output_dir    |
| snapshot_config: interval | interval for saving model parameters         | 5000            |

### Model

| Field             | Usage | Default |
| :---------------------- | -------- | ------ |
| name                    | name of the model | LapStyleRevFirstModel |
| revnet_generator        | set the revnet generator | RevisionNet |
| revnet_discriminator    | set the revnet discriminator | LapStyleDiscriminator |
| draftnet_encode         | set the draftnet encoder | Encoder |
| draftnet_decode         | set the draftnet decoder | DecoderNet |
| calc_style_emd_loss     | set the style loss 1 | CalcStyleEmdLoss |
| calc_content_relt_loss  | set the content loss 1 | CalcContentReltLoss |
| calc_content_loss       | set the content loss 2 | CalcContentLoss |
| calc_style_loss         | set the style loss 2 | CalcStyleLoss |
| gan_criterion: name     | set the GAN loss |  GANLoss |
| gan_criterion: gan_mode | set the modal parameter of GAN loss | vanilla |
| content_layers          | set the network layer that calculates content loss 2 |['r11', 'r21', 'r31', 'r41', 'r51']|
| style_layers            | set the network layer that calculates style loss 2 | ['r11', 'r21', 'r31', 'r41', 'r51'] |
| content_weight          | set the weight of total content loss | 1.0 |
| style_weigh             | set the weight of total style loss | 3.0 |

### Dataset (train & test)

| Field   | Usage           | Default       |
| :----------- | -------------------- | -------------------- |
| name         | name of the dataset | LapStyleDataset      |
| content_root | path of the dataset | data/coco/train2017/ |
| style_root   | path of the target style image | data/starrynew.png   |
| load_size    | image size after resizing the input image | 280          |
| crop_size    | image size after random cropping | 256           |
| num_workers  | number of worker process                         | 16                   |
| batch_size   | size of the data sample for one training session | 5                    |

### Lr_scheduler 

| Field | Usage | Default  |
| :------------ | ---------------- | -------------- |
| name          | name of the learning strategy | NonLinearDecay |
| learning_rate | initial learning rate | 1e-4           |
| lr_decay      | decay rate of the learning rate | 5e-5           |

### Optimizer

| Field | Usage  | Default |
| :-------- | ---------- | ------- |
| name      | class name of the optimizer | Adam    |
| net_names | the network under the optimizer | net_rev |
| beta1     | set beta1, parameter of the optimizer | 0.9     |
| beta2     | set beta2, parameter of the optimizer | 0.999   |

### Validate

| Field | Usage | Default |
| :------- | ---- | ------ |
| interval | validation interval                    | 500    |
| save_img | whether to save image while validating | false  |

### Log_config

| Field    | Usage | Default |
| :--------------- | ---- | ------ |
| interval         | log printing interval | 10     |
| visiual_interval | interval for saving the generated images during training | 500    |
