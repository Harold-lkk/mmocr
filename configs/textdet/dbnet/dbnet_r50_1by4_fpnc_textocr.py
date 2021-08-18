_base_ = './_base_/dbnet_r50_fpnc_textocr.py'

base_channels = 16
model = dict(
    type='DBNet',
    pretrained='pretrain/faster_resnet50_1by4.pth',
    backbone=dict(
        type='FasterResNet',
        faster=True,
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        base_channels=base_channels,
        stem_channels=base_channels,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch'),
    neck=dict(
        type='FPNC', in_channels=[64, 128, 256, 512], lateral_channels=256))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

custom_imports = dict(
    imports=[
        'mmocr.models.common.backbones.faster_resnet',
        'mmocr.models.textdet.losses.db_loss_spe',
        'mmocr.utils.pavi_progress_logger',
        'mmocr.utils.publish_pavi_model_hook',
    ],
    allow_failed_imports=False)
# custom_hooks = [dict(type='PublishPaviModelHook', upload_path='dbnet/demo')]
