model = dict(
    type='CRNNNet',
    preprocessor=None,
    backbone=dict(type='VeryDeepVgg', leaky_relu=False, input_channels=1),
    encoder=None,
    decoder=dict(type='CRNNDecoder', in_channels=512, rnn_flag=True),
    loss=dict(type='CTCLoss'),
    label_convertor=dict(
        type='CTCConvertor',
        dict_type='DICT90',
        with_unknown=True,
        lower=False),
    pretrained=None)

dataset_type = 'OCRDataset'

train_img_prefix = 'datasets/icdar/text_img/'
train_ann_file = 'datasets/icdar/label_train.txt'

test_img_prefix = 'datasets/icdar/text_img/'
test_ann_file = 'datasets/icdar/label_test.txt'

img_norm_cfg = dict(mean=[0.5], std=[0.5])
train_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale'),
    dict(
        type='ResizeOCR',
        height=32,
        min_width=100,
        max_width=100,
        keep_aspect_ratio=False),
    dict(type='ToTensorOCR'),
    dict(type='NormalizeOCR', **img_norm_cfg),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=['filename', 'resize_shape', 'text', 'valid_ratio']),
]
test_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale'),
    dict(
        type='ResizeOCR',
        height=32,
        min_width=4,
        max_width=None,
        keep_aspect_ratio=True),
    dict(type='ToTensorOCR'),
    dict(type='NormalizeOCR', **img_norm_cfg),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=['filename', 'resize_shape', 'text', 'valid_ratio']),
]

loader = dict(
    type='HardDiskLoader',
    repeat=1,
    parser=dict(
        type='LineStrParser',
        keys=['filename', 'text'],
        keys_idx=[0, 1],
        separator=' '))
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type=dataset_type,
        img_prefix=train_img_prefix,
        ann_file=train_ann_file,
        loader=loader,
        pipeline=train_pipeline,
        test_mode=False),
    val=dict(
        type=dataset_type,
        img_prefix=test_img_prefix,
        ann_file=test_ann_file,
        loader=loader,
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type,
        img_prefix=test_img_prefix,
        ann_file=test_ann_file,
        loader=loader,
        pipeline=test_pipeline,
        test_mode=True))

evaluation = dict(interval=1, metric='acc')
# optimizer
optimizer = dict(type='Adadelta', lr=1.0)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='Fixed')
total_epochs = 6

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# imports
custom_imports = dict(
    imports=[
        'mmocr.models.textrecog.losses.spe_ctc_loss.py',
    ],
    allow_failed_imports=False)
