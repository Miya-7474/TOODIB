_base_ = './tood_r50_fpn_coodatt20_ti10_alpha05beta2_ms-2x_coco.py'

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))

# # 获取候选框才用
# test_cfg = dict(
#     nms_pre=1000,
#     min_bbox_size=0,
#     score_thr=0.05,
#     nms=None,
#     max_per_img=1000)

