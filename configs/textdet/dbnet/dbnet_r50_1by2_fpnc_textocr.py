_base_ = './_base_/dbnet_r50_fpnc_textocr.py'

base_channels = 32
model = dict(
    type='DBNet',
    pretrained='pretrain/faster_resnet50_1by2.pth',
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
        type='FPNC', in_channels=[128, 256, 512, 1024], lateral_channels=256))

custom_imports = dict(
    imports=[
        'mmocr.models.common.backbones.faster_resnet',
        'mmocr.models.textdet.losses.db_loss_spe',
        'mmocr.utils.pavi_progress_logger',
        'mmocr.utils.publish_pavi_model_hook',
    ],
    allow_failed_imports=False)
# custom_hooks = [dict(type='PublishPaviModelHook', upload_path='dbnet/demo')]
