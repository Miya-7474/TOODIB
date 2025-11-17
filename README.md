# TOODIB
TOODIB: Task-aligned one-stage object detection with interactions between branches

| Model   | backbone              | mAP  | AP<sub>50 | AP<sub>75 | AP<sub>S | AP<sub>M | AP<sub>L |                                                                                                                                                  Download                                                                                                                                                   |
|---------|-----------------------|:----:|:---------:|:---------:|:--------:|:--------:|:--------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| TOODIB  | ResNet-50             | 43.3 |   59.6    |   47.2    |   25.4   |   46.1   |   57.2   |                                                                                            [config](configs/toodib_DCNv2_4_coordatt_20_ti_10/tood_r50_fpn_DCNv2_4_coodatt_20_ti_10_alpha1beta6_1x_coco.py) / [checkpoint]()                                                                                 |
| TOODIB  | ResNet-101            | 47.6 |   65.1    |   51.9    |   28.9   |   50.8   |   58.5   |                                                                          [config](configs/toodib_DCNv2_4_coordatt_20_ti_10/tood_r101_fpn_DCNv2_4_coodatt_20_ti_10_alpha1beta6_ms-2x_coco.py) / [checkpoint]()                                                                          |
| TOODIB  | ResNeXt-101-64×4d     | 49.0 |   66.4    |   53.4    |   31.0   |   52.3   |   59.7   |                                                                       [config](configs/toodib_DCNv2_4_coordatt_20_ti_10/tood_x101-64x4d_fpn_DCNv2_4_coodatt_20_ti_10_alpha1beta6_ms-2x_coco.py) / [checkpoint]()                                                                       |
| TOODIB  | ResNeXt-101-64×4d-DCN | 50.6 |   68.4    |   55.0    |   31.8   |   53.9   |   63.4   |                                                                 [config](configs/toodib_DCNv2_4_coordatt_20_ti_10/tood_x101-64x4d-dconv-c4-c5_fpn_DCNv2_4_coodatt_20_ti_10_alpha1beta6_ms-2x_coco.py) / [checkpoint]()                                                                 |


## 🔧Installation

1. Clone the repository locally:

    ```shell
    git clone https://github.com/Miya-7474/TOODIB.git
    cd TOODIB/
    ```

2. Create a conda environment and activate it:

    ```shell
    conda create -n toodib python=3.8
    conda activate toodib
    ```

3. Install PyTorch and Torchvision following the instruction on [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/). The code requires `python>=3.8, torch>=1.12.1, torchvision>=0.13.1`.

    ```shell
    conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
    ```

4. Install mmdetection:

    ```shell
    pip install -U openmim
    mim install mmengine
    mim install mmcv==2.1.0
    pip install -v -e .
    ```


## 📁Prepare Dataset

Please download [COCO 2017](https://cocodataset.org/) or prepare your own datasets into `data/`, and organize them as following. 

```shell
data/    
  ├──coco/
      ├── train2017/
      ├── val2017/
      └── annotations/
        ├── instances_train2017.json
        └── instances_val2017.json
```



## 📚︎Train a model
To train a model with one or more GPUs, specify `CUDA_VISIBLE_DEVICES`, `model`, `gpu_num`.
```shell
CUDA_VISIBLE_DEVICES=0 PORT=29500 ./tools/dist_train.sh /configs/toodib_DCNv2_4_coordatt_20_ti_10/tood_r50_fpn_DCNv2_4_coodatt_20_ti_10_alpha1beta6_1x_coco.py 1   # train with 1 GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh /configs/toodib_DCNv2_4_coordatt_20_ti_10/tood_r50_fpn_DCNv2_4_coodatt_20_ti_10_alpha1beta6_1x_coco.py 4  # train with 4 GPUs
```


## 📈Evaluation/Test
To evaluate a model with one or more GPUs, specify `CUDA_VISIBLE_DEVICES`, `model`, `checkpoint`, `gpu_num`.
```shell
CUDA_VISIBLE_DEVICES=<gpu_ids> ./tools/dist_test.sh /path/to/model.py /path/to/checkpoint.pth gpu_num
```



## Other operations
If you want to do other operations such as inference or training with your datasets, please invite the [MMdetection](https://mmdetection.cn/en/latest/user_guides/index.html#) 

## Reference
```bibtex
@article{CHEN2025104567,
title = {TOODIB: Task-aligned one-stage object detection with interactions between branches},
journal = {Computer Vision and Image Understanding},
volume = {262},
pages = {104567},
year = {2025},
issn = {1077-3142},
doi = {https://doi.org/10.1016/j.cviu.2025.104567},
url = {https://www.sciencedirect.com/science/article/pii/S1077314225002905},
author = {Simin Chen and Qinxia Hu and Mingjin Zhu and Qiming Wu and Xiao Hu},

```


