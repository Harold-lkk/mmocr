# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

from torch import Tensor

from mmocr.registry import MODELS, TASK_UTILS
from mmocr.structures import TextRecogDataSample  # noqa F401
from mmocr.utils import DetSampleList, OptMultiConfig
from mmocr.utils.data_sample_convert import (instance_data2recog,
                                             merge_recog2spotting)
from .base import BaseRoIHead


@MODELS.register_module()
class OnlyRecRoIHead(BaseRoIHead):
    """Simplest base roi head including one bbox head and one mask head."""

    def __init__(self,
                 sampler: OptMultiConfig = None,
                 roi_extractor: OptMultiConfig = None,
                 rec_head: OptMultiConfig = None,
                 postprocessor=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if sampler is not None:
            self.sampler = TASK_UTILS.build(sampler)
        self.roi_extractor = MODELS.build(roi_extractor)
        self.rec_head = MODELS.build(rec_head)

    def loss(self, inputs: Tuple[Tensor], data_samples: DetSampleList) -> dict:
        """Perform forward propagation and loss calculation of the detection
        roi on the features of the upstream network.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            DetSampleList (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components
        """

        # assign gts and sample proposals
        proposals = [
            ds.gt_instances[~ds.gt_instances.ignored] for ds in data_samples
        ]

        proposals = [p for p in proposals if len(p) > 0]
        bbox_feats = self.roi_extractor(inputs, proposals)
        rec_data_samples = instance_data2recog(proposals, self.training)
        rec_loss = self.rec_head.loss(bbox_feats, rec_data_samples)

        return rec_loss

    def predict(self, inputs: Tuple[Tensor],
                data_samples: DetSampleList) -> DetSampleList:

        pred_instances = [ds.pred_instances for ds in data_samples]
        bbox_feats = self.roi_extractor(inputs, pred_instances)
        if bbox_feats.size(0) == 0:
            return []
        rec_data_samples = instance_data2recog(pred_instances)
        rec_predicts = self.rec_head.predict(bbox_feats, rec_data_samples)
        data_samples = merge_recog2spotting(rec_predicts, data_samples)
        return data_samples

    def forward(self, inputs, data_samples):
        pred_instances = [ds.pred_instances for ds in data_samples]
        bbox_feats = self.roi_extractor(inputs, pred_instances)
        rec_data_samples = instance_data2recog(pred_instances)
        return self.rec_head(bbox_feats, rec_data_samples)
