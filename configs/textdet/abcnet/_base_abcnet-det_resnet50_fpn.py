file_client_args = dict(backend='disk')
num_classes = 1
strides = [8, 16, 32, 64, 128]
bbox_coder = dict(type='mmdet.DistancePointBBoxCoder')
with_bezier = True
norm_on_bbox = True
use_sigmoid_cls = True
model = dict(
    type='ABCNetDet',
    data_preprocessor=dict(
        type='TextDetDataPreprocessor',
        mean=[123.675, 116.28, 103.53][::-1],
        std=[1, 1, 1],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe')),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_output',  # use P5
        num_outs=6,
        relu_before_extra_convs=True),
    det_head=dict(
        type='ABCDetHead',
        num_classes=num_classes,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=strides,
        norm_on_bbox=norm_on_bbox,
        use_sigmoid_cls=use_sigmoid_cls,
        centerness_on_reg=True,
        dcn_on_last_conv=False,
        conv_bias=True,
        use_scale=False,
        with_bezier=with_bezier,
        init_cfg=dict(
            type='Normal',
            layer='Conv2d',
            std=0.01,
            override=dict(
                type='Normal',
                name='conv_cls',
                std=0.01,
                bias=-4.59511985013459),  # -log((1-p)/p) where p=0.01
        ),
        module_loss=dict(
            type='ABCNetDetModuleLoss',
            num_classes=num_classes,
            strides=strides,
            center_sampling=True,
            center_sample_radius=1.5,
            bbox_coder=bbox_coder,
            with_bezier=with_bezier,
            norm_on_bbox=norm_on_bbox,
            use_sigmoid_cls=use_sigmoid_cls,
            loss_cls=dict(
                type='mmdet.FocalLoss',
                use_sigmoid=use_sigmoid_cls,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(type='mmdet.GIoULoss', loss_weight=1.0),
            loss_centerness=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=True,
                loss_weight=1.0)),
        postprocessor=dict(
            type='ABCNetDetPostprocessor',
            rescale_fields=['polygons', 'bboxes', 'beziers'],
            use_sigmoid_cls=True,
            strides=[8, 16, 32, 64, 128],
            bbox_coder=dict(type='mmdet.DistancePointBBoxCoder'),
            with_bezier=True,
            test_cfg=dict(
                # rescale_fields=['polygon', 'bboxes', 'bezier'],
                nms_pre=1000,
                nms=dict(type='nms', iou_threshold=0.5),
                score_thr=0.3))))

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args=file_client_args,
        color_type='color_ignore_orientation'),
    dict(type='Resize', scale=(2000, 4000), keep_ratio=True, backend='pillow'),
    dict(
        type='LoadOCRAnnotations',
        with_polygon=True,
        with_bbox=True,
        with_label=True),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

train_pipeline = test_pipeline
