_base_ = []
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

label_convertor = dict(
    type='CTCConvertor', dict_type='DICT90', with_unknown=True, lower=False)

model = dict(
    type='CRNNNet',
    preprocessor=None,
    backbone=dict(type='VeryDeepVgg', leaky_relu=False, input_channels=1),
    encoder=None,
    decoder=dict(type='CRNNDecoder', in_channels=512, rnn_flag=True),
    loss=dict(type='CTCLoss'),
    label_convertor=label_convertor,
    pretrained=None)

train_cfg = None
test_cfg = None

# optimizer
optimizer = dict(type='Adadelta', lr=1.0)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='Fixed', warmup='linear', warmup_iters=100, warmup_ratio=0.1)
total_epochs = 6

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
        meta_keys=[
            'filename', 'ori_shape', 'img_shape', 'text', 'valid_ratio'
        ]),
]
train_pipeline_ceph = [
    dict(
        type='LoadImageFromFile',
        file_client_args=dict(backend='petrel'),
        color_type='grayscale'),
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
        meta_keys=['filename', 'ori_shape', 'img_shape', 'valid_ratio']),
]

dataset_type = 'OCRDataset'

train_prefix = 'data/mixture/'

train_img_prefix1 = '/mnt/lustre/share_data/openmmlab/datasets/mmocr/recog/ \
                        SynthText/synthtext/SynthText_patch_horizontal'

train_img_prefix2 = 's3://yuexiaoyu.ocr_dataset/mnt/ramdisk/max/90kDICT32px'

ann_prefix = '/mnt/lustre/share_data/openmmlab/datasets/mmocr/recog/testset/'
train_ann_file1 = ann_prefix + 'SynthText/synthtext/full_labels.lmdb'
train_ann_file2 = ann_prefix + 'mnt/ramdisk/max/90kDICT32px/full_labels.lmdb'

train1 = dict(
    type=dataset_type,
    img_prefix=train_img_prefix1,
    ann_file=train_ann_file1,
    loader=dict(
        type='LmdbLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=train_pipeline,
    test_mode=False)

train2 = {key: value for key, value in train1.items()}
train2['img_prefix'] = train_img_prefix2
train2['ann_file'] = train_ann_file2
train2['pipeline'] = train_pipeline_ceph

test_prefix = '/mnt/lustre/share_data/openmmlab/datasets/mmocr/ \
                recog/testset/testset/'

test_img_prefix1 = test_prefix + 'IIIT5K/'
test_img_prefix2 = test_prefix + 'svt/'
test_img_prefix3 = test_prefix + 'icdar_2013/Challenge2_Test_Task3_Images/'
test_img_prefix4 = test_prefix + 'icdar_2015/ch4_test_word_images_gt/'
test_img_prefix5 = test_prefix + 'svtp/'
test_img_prefix6 = test_prefix + 'ct80/'

test_ann_file1 = test_prefix + 'IIIT5K/label.txt'
test_ann_file2 = test_prefix + 'svt/test_list.txt'
test_ann_file3 = test_prefix + 'icdar_2013/1015_test_label.txt'
test_ann_file4 = test_prefix + 'icdar_2015/test_label.txt'
test_ann_file5 = test_prefix + 'svtp/imagelist.txt'
test_ann_file6 = test_prefix + 'ct80/imagelist.txt'

test1 = dict(
    type=dataset_type,
    img_prefix=test_img_prefix1,
    ann_file=test_ann_file1,
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=True)

test2 = {key: value for key, value in test1.items()}
test2['img_prefix'] = test_img_prefix2
test2['ann_file'] = test_ann_file2

test3 = {key: value for key, value in test1.items()}
test3['img_prefix'] = test_img_prefix3
test3['ann_file'] = test_ann_file3

test4 = {key: value for key, value in test1.items()}
test4['img_prefix'] = test_img_prefix4
test4['ann_file'] = test_ann_file4

test5 = {key: value for key, value in test1.items()}
test5['img_prefix'] = test_img_prefix5
test5['ann_file'] = test_ann_file5

test6 = {key: value for key, value in test1.items()}
test6['img_prefix'] = test_img_prefix6
test6['ann_file'] = test_ann_file6

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='UniformConcatDataset', datasets=[train1, train2], pipeline=None),
    val=dict(
        type='UniformConcatDataset',
        datasets=[test1, test2, test3, test4, test5, test6],
        pipeline=test_pipeline),
    test=dict(
        type='UniformConcatDataset',
        datasets=[test1, test2, test3, test4, test5, test6],
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='acc')
