# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.transforms import LoadImageFromFile
from mmengine.dataset.sampler import DefaultSampler

from mmdet.datasets import AspectRatioBatchSampler, CocoDataset
from mmdet.datasets.transforms import (LoadAnnotations, PackDetInputs,
                                       RandomFlip, Resize)
from mmdet.evaluation import CocoMetric

# dataset settings
dataset_type = CocoDataset
data_root = 'data/coco/'

# Example to use different file client使用不同的文件客户端的示例
# 方法 1: simply set the data root and let the file I/O module简单设置数据根目录并让文件I/O模块自动推断前缀
# automatically infer from prefix (not support LMDB and Memcache yet)# 从前缀自动推断（尚不支持LMDB和Memcache）

# data_root = 's3://openmmlab/datasets/detection/coco/'

# 方法 2: 使用 `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
backend_args = None

train_pipeline = [# 训练数据处理流程
    dict(type=LoadImageFromFile, backend_args=backend_args),# 第 1 个流程，从文件路径里加载图像。
    dict(type=LoadAnnotations, with_bbox=True), # 第 2 个流程，对于当前图像，加载它的注释信息。
    dict(type=Resize,
         scale=(1333, 800),# 图像的最大尺寸
         keep_ratio=True),# 是否保持图像的长宽比
    dict(type=RandomFlip, prob=0.5),# 翻转图像和其标注的数据增广流程。概率50%
    dict(type=PackDetInputs)# 将数据转换为检测器输入格式的流程
]
test_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=Resize, scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline# 如果没有 gt 注释，删除流程
    dict(type=LoadAnnotations, with_bbox=True),
    dict(
        type=PackDetInputs,# 将数据转换为检测器输入格式的流程
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(# 训练 dataloader 配置
    batch_size=2,# 单个 GPU 的 batch size
    num_workers=2,# 单个 GPU 分配的数据加载线程数
    persistent_workers=True,# 如果设置为 True，dataloader 在迭代完一轮之后不会关闭数据读取的子进程，可以加速训练
    sampler=dict(type=DefaultSampler, shuffle=True),# 训练数据的采样器
    batch_sampler=dict(type=AspectRatioBatchSampler),# 批数据采样器，用于确保每一批次内的数据拥有相似的长宽比，可用于节省显存
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',# 标注文件路径
        data_prefix=dict(img='train2017/'),# 图片路径前缀
        filter_cfg=dict(filter_empty_gt=True, min_size=32),# 图片和标注的过滤配置
        pipeline=train_pipeline,# 这是由之前创建的 train_pipeline 定义的数据处理流程。
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=1,#如果 batch_size > 1，组成 batch 时的额外填充会影响模型推理精度
    num_workers=2,
    persistent_workers=True,
    drop_last=False,# 是否丢弃最后未能组成一个批次的数据
    sampler=dict(type=DefaultSampler, shuffle=False),# 验证和测试时不打乱数据顺序
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,# 开启测试模式，避免数据集过滤图片和标注
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(# 验证过程使用的评测器
    type=CocoMetric,# 用于评估检测和实例分割的 AR、AP 和 mAP 的 coco 评价指标
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric='bbox',# 需要计算的评价指标，`bbox` 用于检测，`segm` 用于实例分割
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator

# inference on test dataset and
# format the output results for submission.
# test_dataloader = dict(
#     batch_size=1,
#     num_workers=2,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type=DefaultSampler, shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file=data_root + 'annotations/image_info_test-dev2017.json',
#         data_prefix=dict(img='test2017/'),
#         test_mode=True,
#         pipeline=test_pipeline))
# test_evaluator = dict(
#     type=CocoMetric,
#     metric='bbox',
#     format_only=True,
#     ann_file=data_root + 'annotations/image_info_test-dev2017.json',
#     outfile_prefix='./work_dirs/coco_detection/test')
