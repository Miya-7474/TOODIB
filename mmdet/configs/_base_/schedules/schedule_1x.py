# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.optim.optimizer.optimizer_wrapper import OptimWrapper
from mmengine.optim.scheduler.lr_scheduler import LinearLR, MultiStepLR
from mmengine.runner.loops import EpochBasedTrainLoop, TestLoop, ValLoop
from torch.optim.sgd import SGD

# training schedule for 1x
train_cfg = dict(type=EpochBasedTrainLoop, # 训练循环的类型，main/mmengine/runner/loops.py
                 max_epochs=12, # 最大训练轮次
                 val_interval=1)# 验证间隔。每个 epoch 验证一次
val_cfg = dict(type=ValLoop)
test_cfg = dict(type=TestLoop)

# learning rate
param_scheduler = [#配置参数调度器（Parameter Scheduler）来调整优化器的超参数
    dict(type=LinearLR, # 使用线性学习率预热
         start_factor=0.001,# 学习率预热的系数
         by_epoch=False,# 按 iteration 更新预热学习率
         begin=0,# 从第一个 iteration 开始
         end=500),# 在第 500 个 iteration 结束
    dict(
        type=MultiStepLR,# 在训练过程中使用 multi step 学习率策略
        begin=0,
        end=12,
        by_epoch=True,# 按 epoch 更新学习率
        milestones=[8, 11],# 在哪几个 epoch 进行学习率衰减
        gamma=0.1)# 学习率衰减系数
]

# optimizer
optim_wrapper = dict(# 优化器封装的配置
    type=OptimWrapper,# 优化器封装的类型。可以切换至 AmpOptimWrapper 来启用混合精度训练
    optimizer=dict(type=SGD, lr=0.02, momentum=0.9, weight_decay=0.0001))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
