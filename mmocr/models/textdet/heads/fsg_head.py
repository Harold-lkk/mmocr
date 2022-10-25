# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from mmocr.registry import MODELS
from .base import BaseTextDetHead


@MODELS.register_module()
class FSGHead(BaseTextDetHead):
    """The class for implementing FSGNet text detector:

    Few Could Be Better Than All: Feature Sampling and Grouping for Scene Text
    Detection [https://arxiv.org/abs/2203.15221v2].
    """

    def __init__(
        self,
        in_channels: int = 256,
        top_n: List[int] = [256, 128, 64],
        module_loss: Dict = dict(type='DBModuleLoss'),
        encoder=dict(
            type='TransformerLayerSequence',
            num_layers=4,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=512,
                    num_heads=8,
                    dropout=0.1),
                ffn_cfgs=dict(
                    type='FFN',
                    embed_dims=512,
                    feedforward_channels=1024,
                    num_fcs=2,
                    ffn_drop=0.0,
                    act_cfg=dict(type='ReLU', inplace=True),
                ),
                operation_order=(
                    'self_attn',
                    'norm',
                    'cross_attn',
                    'norm',
                    'ffn',
                    'norm',
                ),
            ),
        ),
        postprocessor: Dict = dict(type='DBPostprocessor'),
        init_cfg: Optional[Union[Dict, List[Dict]]] = None,
    ) -> None:
        super().__init__(module_loss, postprocessor, init_cfg)
        self.top_n = top_n
        self.pooling = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.sample = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid())
        self.encoder = MODELS.build(encoder)
        self.cls = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())
        self.box = nn.Sequential(nn.Linear(512, 5), nn.Sigmoid())
        self.row_embedding = nn.Embedding(64, 128)
        self.col_embedding = nn.Embedding(64, 128)
        self.flatten = nn.Flatten()

    def forward(self, feats: List[Tensor], data_sample=None) -> List[Tensor]:
        """Forward function.

        Args:
            feats (List[Tensor]): List of feature maps.

        Returns:
            List[Tensor]: List of feature maps.
        """
        score_maps, top_feats = list(
            zip(*map(self._forward_single_sampling, feats[:-1], self.top_n)))

        top_feats = torch.cat(top_feats, dim=-1)

        bboxes, classes = self._forward_grouping(top_feats)
        return score_maps, bboxes, classes

    def _forward_single_sampling(self, feat: Tensor, top_n: int) -> Tensor:
        """"""
        feat = self.pooling(feat)
        score_feat = self.sample(feat)
        _, indices = torch.topk(self.flatten(score_feat), top_n)
        h, w = score_feat.shape[-2:]
        col_coord = indices % w
        row_coord = indices // w
        top_feat = torch.stack([
            feat[i, :, row_coord[i], col_coord[i]]
            for i in range(feat.shape[0])
        ])
        row_emb = self.row_embedding(col_coord)
        col_emb = self.col_embedding(col_coord)
        position_feat = torch.cat([row_emb, col_emb], dim=-1).permute(0, 2, 1)
        top_feat = torch.cat([top_feat, position_feat], dim=1)
        return score_feat, top_feat

    def _forward_grouping(self, top_feats):
        top_feats = top_feats.permute(2, 0, 1)
        top_feats = self.encoder(top_feats, None, None)
        top_feats = top_feats.permute(1, 0, 2)
        score_map = self.cls(top_feats)
        box_map = self.box(top_feats)
        return box_map, score_map
