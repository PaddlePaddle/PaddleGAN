English (./README.md)

# Usage

To compute the FID score between two datasets, where images of each dataset are contained in an individual folder:

Download [InceptionV3](https://paddlegan.bj.bcebos.com/InceptionV3.pdparams) inference model, and then run
```
python test_fid_score.py --image_data_path1 /path/to/dataset1 --image_data_path2 /path/to/dataset2 --inference_model /path/to/InceptionV3/inference_moel
```
