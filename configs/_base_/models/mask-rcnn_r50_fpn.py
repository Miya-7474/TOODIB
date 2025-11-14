# model settings
model = dict(
    type='MaskRCNN',

    data_preprocessor=dict(# 数据预处理器的配置，通常包括图像归一化和 padding
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],# 用于预训练骨干网络的图像归一化通道均值，按 R、G、B 排序
        std=[58.395, 57.12, 57.375],# 标准差，按 R、G、B 排序
        bgr_to_rgb=True,# 从BGR转为RGB
        pad_mask=True,# 填充实例分割掩码
        pad_size_divisor=32),# padding 后的图像的大小应该可以被 ``pad_size_divisor`` 整除

    backbone=dict(
        type='ResNet',
        depth=50,# 主干网络的深度，resnet50
        num_stages=4,# 主干网络状态(stages)的数目
        out_indices=(0, 1, 2, 3),# 每个状态产生的特征图输出的索引
        frozen_stages=1,# 第一个状态冻结
        norm_cfg=dict(type='BN', requires_grad=True),# 归一化层
        norm_eval=True,# 冻结BN里的统计项
        style='pytorch',# 主干网络的风格,'pytorch'意思是步长为2的层为 3x3 卷积,'caffe'意思是步长为2的层为 1x1 卷积
        init_cfg=dict(# 加载设置，预训练的模型
            type='Pretrained', checkpoint='torchvision://resnet50')),

    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],# 输入通道数，对应主干网络的输出
        out_channels=256,# 金字塔特征图每一层的输出通道
        num_outs=5),# 输出的范围

    rpn_head=dict(
        type='RPNHead',
        in_channels=256, # 每个输入特征图的输入通道，对应neck的输出
        feat_channels=256,# head卷积层的特征通道

        anchor_generator=dict(# Anchor生成器的配置
            type='AnchorGenerator',
            scales=[8],# 锚点的基本比例，特征图某一位置的锚点面积为 scale * base_sizes
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),# 锚生成器的步幅。这与 FPN 特征步幅一致。

        bbox_coder=dict(# 在训练和测试期间对框进行编码和解码
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],# 用于编码和解码框的目标均值
            target_stds=[1.0, 1.0, 1.0, 1.0]),# 标准差

        loss_cls=dict(# 分类分支的损失函数配置
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),

    roi_head=dict(  # RoIHead 封装了两步(two-stage)/级联(cascade)检测器的第二步
        type='StandardRoIHead',

        bbox_roi_extractor=dict(  # 用于bbox回归的RoI特征提取器
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign',  # RoI 层的配置
                           output_size=7,  # 特征图的输出大小
                           sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),  # 多尺度特征图的步幅，应该与主干的架构保持一致

        bbox_head=dict(  # RoIHead中box head的配置
            type='Shared2FCBBoxHead',
            in_channels=256,  # 对应roi_extractor 中的 out_channels
            fc_out_channels=1024,  # FC 层的输出特征通道
            roi_feat_size=7,  # 候选区域(Region of Interest)特征的大小
            num_classes=80,
            bbox_coder=dict(  # 第二阶段使用的框编码器
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),  # 因为框更准确，所以值更小
            reg_class_agnostic=False,  # 回归是否与类别无关
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),

        mask_roi_extractor=dict(# 用于 mask 生成的 RoI 特征提取器
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),

        mask_head=dict(# mask 预测 head 模型
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,# 输入通道，对应mask roi extractor 的输出通道
            conv_out_channels=256,
            num_classes=80,
            loss_mask=dict(# mask 分支的损失函数配置
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),


    # model training and testing settings
    train_cfg=dict(  # rpn 和 rcnn 训练超参数的配置

        rpn=dict(
            assigner=dict(  # 分配器(assigner)的配置
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,  # IoU >= 0.7(阈值) 被视为正样本
                neg_iou_thr=0.3,  # IoU < 0.3 被视为负样本
                min_pos_iou=0.3,  # 将框作为正样本的最小 IoU 阈值
                match_low_quality=True,  # 是否匹配低质量的框
                ignore_iof_thr=-1),  # 忽略 bbox 的 IoF 阈值
            sampler=dict(  # 正/负采样器的配置
                type='RandomSampler',
                num=256,  # 样本数量。
                pos_fraction=0.5,  # 正样本占总样本的比例
                neg_pos_ub=-1,  # 基于正样本数量的负样本上限
                add_gt_as_proposals=False),  # 采样后是否添加 GT 作为 proposal
            allowed_border=-1,  # 填充有效锚点后允许的边框
            pos_weight=-1,  # 训练期间正样本的权重
            debug=False),

        rpn_proposal=dict(  # 在训练期间生成 proposals 的配置
            nms_pre=2000,  # NMS 前的 box 数
            max_per_img=1000,  # NMS 后保留的 box 的数量
            nms=dict(type='nms', iou_threshold=0.7),  # NMS 的配置
            min_bbox_size=0),  # 允许的最小 box 尺寸

        rcnn=dict(  # roi head 的配置
            assigner=dict(  # 第二阶段分配器的配置，这与 rpn 中的不同
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),  # 采样后添加 GT 作为 proposal
            mask_size=28,  # mask 的大小
            pos_weight=-1,
            debug=False)),

    test_cfg=dict(  # 用于测试 rpn 和 rcnn 超参数的配置

        rpn=dict(  # 测试阶段生成 proposals 的配置
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),

        rcnn=dict(  # roi heads 的配置
            score_thr=0.05,  # bbox 的分数阈值
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,  # 每张图像的最大检测次数
            mask_thr_binary=0.5)))  # mask 预处的阈值
