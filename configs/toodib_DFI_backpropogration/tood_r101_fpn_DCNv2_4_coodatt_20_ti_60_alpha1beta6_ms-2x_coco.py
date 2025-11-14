_base_ = './tood_r50_fpn_DCNv2_4_coodatt_20_ti_60_alpha1beta6_ms-2x_coco.py'

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101'))
)
