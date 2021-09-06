# optimizer
# optimizer = dict(type='SGD', lr=0.007, momentum=0.9, weight_decay=0.0001)
optimizer = dict(type='Adam', lr=0.001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-7, by_epoch=True)
total_epochs = 100

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook')
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

model = dict(
    type='DBNet',
    # pretrained='pretrain/resnet50-19c8e357.pth',
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch'),
    neck=dict(
        type='FPNC', in_channels=[256, 512, 1024, 2048], lateral_channels=256),
    bbox_head=dict(
        type='DBHead',
        text_repr_type='poly',
        in_channels=256,
        loss=dict(
            type='SpeDBLoss',
            probability_head=dict(type='MaskBalanceBCELoss', loss_weight=5),
            threshold_head=dict(type='MaskL1Loss', loss_weight=10),
            diff_binary_head=dict(type='MaskDiceLoss', loss_weight=1))),
    train_cfg=None,
    test_cfg=None)

dataset_type = 'IcdarDataset'
data_root = 'datasets/icdar2015/imgs/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadTextAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(type='ColorJitter', brightness=32.0 / 255, saturation=0.5),
    dict(type='Normalize', **img_norm_cfg),
    # img aug
    dict(
        type='ImgAug',
        args=[['Fliplr', 0.5],
              dict(cls='Affine', rotate=[-10, 10]), ['Resize', [0.5, 3.0]]]),
    # random crop
    dict(type='EastRandomCrop', target_size=(640, 640)),
    dict(type='DBNetTargets', shrink_ratio=0.4),
    dict(type='Pad', size_divisor=32),
    # for visualizing img and gts, pls set visualize = True
    dict(
        type='CustomFormatBundle',
        keys=['gt_shrink', 'gt_shrink_mask', 'gt_thr', 'gt_thr_mask'],
        visualize=dict(flag=False, boundary_key='gt_shrink')),
    dict(
        type='Collect',
        keys=['img', 'gt_shrink', 'gt_shrink_mask', 'gt_thr', 'gt_thr_mask'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1280, 1280),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(1280, 1280), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    val_dataloader=dict(samples_per_gpu=8),
    test_dataloader=dict(samples_per_gpu=8),
    shuffle=False,
    train=dict(
        type=dataset_type,
        ann_file=data_root + '/textdet_annotations_train.json',
        # for debugging top k imgs
        # select_first_k=50,
        img_prefix=data_root + '/training',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + '/textdet_annotations_test.json',
        img_prefix=data_root + '/training',
        # select_first_k=50,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + '/textdet_annotations_test.json',
        img_prefix=data_root + '/training',
        # select_first_k=100,
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='hmean-iou')

custom_imports = dict(
    imports=[
        'mmocr.models.textdet.losses.db_loss_spe',
        'mmocr.datasets.ocr_e2e_dataset',
        'mmocr.utils.pavi_progress_logger',
        'mmocr.utils.publish_pavi_model_hook',
    ],
    allow_failed_imports=False)

# custom_hooks = [
#     dict(type='PublishPaviModelHook', upload_path='checkpoints/demo')
# ]
