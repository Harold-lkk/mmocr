# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence  # noqa

import torch
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from mmengine.structures import InstanceData
from torch import Tensor

from mmocr.registry import MODELS, TASK_UTILS
from mmocr.structures import TextDetDataSample
# rbox convert
# from mmrotate import xxx
from mmocr.utils.typing import ConfigType, DetSampleList
from .base import BaseTextDetModuleLoss


@MODELS.register_module()
class FSGModuleLoss(BaseTextDetModuleLoss):

    def __init__(
        self,
        loss_score_map=dict(type='MaskedSmoothL1Loss'),
        loss_cls: ConfigType = dict(type='MaskedBCELoss'),
        loss_bbox: ConfigType = dict(type='mmrotate.GWDLoss', loss_weight=5.0),
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='ClassificationCost', weight=1.0),
                dict(type='GWDCost', weight=5.0, box_format='xywh'),
            ],
        ),
    ) -> None:
        super().__init__()
        self.scope_map = MODELS.build(loss_score_map)
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.assigner = TASK_UTILS.build(assigner)

    def _get_reg_cls_targets_single(self, cls_score: Tensor, bbox_pred: Tensor,
                                    data_smaple: TextDetDataSample) -> tuple:
        gt_instances = data_smaple.gt_instances
        img_h, img_w = data_smaple.get('img_shape')
        factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        num_bboxes = bbox_pred.size(0)
        # convert bbox_pred from xywh, normalized to xyxy, unnormalized
        bbox_pred = bbox_cxcywh_to_xyxy(bbox_pred)
        bbox_pred = bbox_pred * factor
        pred_instances = InstanceData(scores=cls_score, bboxes=bbox_pred)
        # assigner and sampler
        assign_result = self.assigner.assign(
            pred_instances=pred_instances,
            gt_instances=gt_instances,
            img_meta=data_smaple.metainfo,
        )
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        pos_inds = (
            torch.nonzero(assign_result.gt_inds > 0,
                          as_tuple=False).squeeze(-1).unique())
        neg_inds = (
            torch.nonzero(assign_result.gt_inds == 0,
                          as_tuple=False).squeeze(-1).unique())
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds.long(), :]

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        pos_gt_bboxes_normalized = pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)

    def _get_score_targets_single_level(self, polygons, img_shape) -> tuple:
        pass

    def _get_score_targets_single(self,
                                  data_sample: TextDetDataSample) -> tuple:
        pass

    def forward(self, preds: List[Tensor], data_sample: DetSampleList) -> dict:
        """Forward function.

        Args:
            preds (List[Tensor]): List of feature maps. score_map, cls, bbox
            data_sample (DetSampleList): List of data sample.

        Returns:
            dict: Loss dict.
        """
        loss_dict = dict()
        return loss_dict
