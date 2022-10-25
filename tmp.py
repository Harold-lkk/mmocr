import torch

from mmocr.models.textdet import FCENet
from mmocr.structures import TextDetDataSample
from mmocr.utils import register_all_modules

register_all_modules()
model = FCENet(
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=4),
    det_head=dict(type='FSGHead'),
)

inputs = torch.rand(1, 3, 512, 512)
model(inputs, [TextDetDataSample()])
