dictionary = dict(
    type='Dictionary',
    dict_file='{{ fileDirname }}/../../../dicts/abcnet.txt',
    with_start=False,
    with_end=False,
    same_start_end=False,
    with_padding=True,
    with_unknown=True)

model = dict(
    type='ABCNetRec',
    data_preprocessor=dict(
        type='TextRecogDataPreprocessor', mean=[0, 0, 0], std=[1, 1, 1]),
    backbone=dict(type='ABCRecBackbone'),
    encoder=dict(type='ABCEncoder'),
    decoder=dict(
        type='ABCDecoder',
        dictionary=dictionary,
        postprocessor=dict(type='AttentionPostprocessor'),
        module_loss=dict(
            type='CEModuleLoss', ignore_first_char=True, reduction='mean'),
        max_seq_len=25),
)

file_client_args = dict(backend='disk')
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args=file_client_args,
        ignore_empty=True,
        min_size=2),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(
        type='RescaleToHeight',
        height=48,
        min_width=48,
        max_width=160,
        width_divisor=4),
    dict(type='PadToWidth', width=160),
    dict(
        type='PackTextRecogInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
]

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(
        type='RescaleToHeight',
        height=48,
        min_width=48,
        max_width=160,
        width_divisor=4),
    dict(type='PadToWidth', width=160),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(
        type='PackTextRecogInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
]
