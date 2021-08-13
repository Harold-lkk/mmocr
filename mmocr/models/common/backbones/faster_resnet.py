import torch.nn as nn
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmdet.models.backbones.resnet import ResNet
from mmdet.models.builder import BACKBONES


@BACKBONES.register_module()
class FasterResNet(ResNet):
    """Faster ResNet backbone.

    Args:
        faster (bool): A flag to change conv1 from 7X7 to 3X3. Default: True.
        ceil_mode (bool): A flag to use ceil mode in the max pooling layer.
        max_pool_cfg (dict): The config for the max pooling layer. It cannot
          co-exist with ceil_mode. Default: None.
    """

    def __init__(self,
                 *args,
                 faster=True,
                 ceil_mode=False,
                 max_pool_cfg=None,
                 **kwargs):
        self.faster = faster
        self.ceil_mode = ceil_mode
        self.max_pool_cfg = max_pool_cfg
        if max_pool_cfg is not None and ceil_mode:
            raise ValueError(
                'ceil_mode and max_pool_cfg cannot be assigned simultaneously')
        super().__init__(*args, **kwargs)

    def _make_stem_layer(self, in_channels, base_channels):
        """Most implementation is the same with :class:`ResNet`.

        conv1 is allowed to change from 7x7 to 3x3 if `self.faster` is True.
        maxpool is allowed to change to ceil_mode if `self.ceil_mode` is True.
        """
        if self.deep_stem:
            # No change in deep_stem
            super()._make_stem_layer(in_channels, base_channels)
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                base_channels,
                kernel_size=3 if self.faster else 7,
                stride=2,
                padding=1 if self.faster else 3,
                bias=False)
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, base_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)
        if self.ceil_mode:
            self.maxpool = nn.MaxPool2d(
                kernel_size=3, stride=2, padding=0, ceil_mode=True)
        elif self.max_pool_cfg is None:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.maxpool = nn.MaxPool2d(**self.max_pool_cfg)
