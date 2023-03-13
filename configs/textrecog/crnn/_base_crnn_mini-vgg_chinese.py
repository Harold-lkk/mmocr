_base_ = [
    '_base_crnn_mini-vgg.py',
]
dictionary = dict(
    type='Dictionary',
    dict_file='{{ fileDirname }}/../../../dicts/benchmark_line.txt',
    with_padding=True)
_base_.model.decoder.dictionary = dictionary
_base_.model.backbone.input_channels = 3
_base_.model.data_preprocessor.bgr_to_rgb = True
file_client_args = dict(backend='disk')
test_pipeline = [
    dict(type='LoadImageFromNDArray'),
    dict(
        type='RescaleToHeight',
        height=32,
        min_width=256,
        max_width=256,
        width_divisor=16,
        backend='pillow'),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(
        type='PackTextRecogInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
]
