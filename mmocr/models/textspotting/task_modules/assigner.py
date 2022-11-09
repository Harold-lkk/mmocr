# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmocr.registry import TASK_UTILS


# assign 负责将gt和预测的bbox进行匹配
# 负责采样后续的训练样本(gt = xx, pred = xxx)
class BaseAssigner:

    def __init__(self):
        pass

    def assign(self, data_samples):
        pass


@TASK_UTILS.register_module()
class PseudoAssigner(BaseAssigner):

    def assign(self, data_samples):
        pred_instances = data_samples.get('pred_instances', None)
        if pred_instances is not None:
            gt_indices = torch.full(len(pred_instances), -1, dtype=torch.uint8)
            pred_instances.gt_indices = gt_indices
        return data_samples
