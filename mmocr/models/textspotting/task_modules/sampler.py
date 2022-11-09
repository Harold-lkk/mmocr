# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmocr.registry import TASK_UTILS


class BaseSampler:

    def __init__(self, gt_fraction, pred_fraction, assigner=None) -> None:
        self.gt_fraction = gt_fraction
        self.pred_fraction = pred_fraction
        if assigner is not None:
            self.assigner = TASK_UTILS.build(assigner)

    def have_assigner(self):
        return hasattr(self, 'assigner') and self.assigner is not None

    def _sample_pred(self, pred_instances, fraction):
        pass

    def _sample_gt(self, gt_instances, fraction):
        pass

    def sample(self, data_samples):
        if self.have_assigner:
            data_samples = self.assigner.assign(data_samples)
        pred_instances = data_samples.get('pred_instances', None)
        gt_instances = data_samples.gt_instances
        if pred_instances is not None and self.pred_fraction > 0:
            pred_indices = self._sample_pred(pred_instances,
                                             self.pred_fraction)
            select_pred_instances = pred_instances[pred_indices]
            select_pred_instances.text = gt_instances[
                select_pred_instances.gt_inds].text
        gt_instances = data_samples.get('gt_instances', None)
        if self.gt_fraction > 0:
            gt_indices = self._sample_gt(gt_instances, self.gt_fraction)
            select_gt_instances = gt_instances[gt_indices]

        data_samples.proposals = select_gt_instances.cat(
            select_gt_instances, select_pred_instances)
        return data_samples


@TASK_UTILS.register_module()
class OnlyGTSampler(BaseSampler):

    def _sample_gt(self, gt_instances, fraction=1):
        gt_ignored = gt_instances.ignored
        gt_indices = torch.nonzero(~gt_ignored, as_tuple=False)
        return gt_indices
