default_scope = 'mmdet'# 默认的注册器域名，默认从此注册器域中寻找模块。

default_hooks = dict(
    timer=dict(type='IterTimerHook'),#统计迭代耗时
    logger=dict(type='LoggerHook', interval=50),#打印日志
    param_scheduler=dict(type='ParamSchedulerHook'),#调用 ParamScheduler 的 step 方法
    checkpoint=dict(type='CheckpointHook', interval=1),#按指定间隔保存权重
    sampler_seed=dict(type='DistSamplerSeedHook'),#确保分布式 Sampler 的 shuffle 生效
    visualization=dict(type='DetVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,# 是否启用 cudnn benchmark
    mp_cfg=dict(mp_start_method='fork',# 多进程设置使用 fork 来启动多进程。'fork' 通常比 'spawn' 更快，但可能存在隐患。
                opencv_num_threads=0), # 关闭 opencv 的多线程以避免系统超负荷
    dist_cfg=dict(backend='nccl'),# 分布式相关设置
)

vis_backends = [dict(type='LocalVisBackend')]# 可视化后端
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)# 日志处理器用于处理运行时日志,使用 epoch 格式的日志

log_level = 'INFO'# 日志等级
load_from = None# 从给定路径加载模型检查点作为预训练模型。这不会恢复训练。
resume = False# 是否从 `load_from` 中定义的检查点恢复。 如果 `load_from` 为 None，它将恢复 `work_dir` 中的最新检查点。
