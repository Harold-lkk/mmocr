# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.models.task_modules.assigners.match_cost import BaseMatchCost
from mmrotate.models.losses.gaussian_dist_loss import xy_wh_r_2_xy_sigma
from torch import Tensor

from mmocr.registry import TASK_UTILS
from mmocr.structures import TextDetDataSample


@TASK_UTILS.register_module()
class GWDCost(BaseMatchCost):

    def __init__(self, fun, tau=2.0) -> None:
        self.fun = fun
        self.tau = tau

    def __call__(self, data_sample: TextDetDataSample) -> Tensor:
        pred = data_sample.pred_instances.rboxes
        gt = data_sample.gt_instances.rboxes
        mu_p, sigma_p = xy_wh_r_2_xy_sigma(pred)
        mu_t, sigma_t = xy_wh_r_2_xy_sigma(gt)

        xy_distance = (mu_p - mu_t).square().sum(dim=-1)

        whr_distance = sigma_p.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        whr_distance = whr_distance + sigma_t.diagonal(
            dim1=-2, dim2=-1).sum(dim=-1)

        _t_tr = (sigma_p.bmm(sigma_t)).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        _t_det_sqrt = (sigma_p.det() * sigma_t.det()).clamp(0).sqrt()
        whr_distance += (-2) * (_t_tr + 2 * _t_det_sqrt).clamp(0).sqrt()

        dis = xy_distance + whr_distance
        gwd_dis = dis.clamp(min=1e-6)

        if self.fun == 'sqrt':
            loss = 1 - 1 / (self.tau + torch.sqrt(gwd_dis))
        elif self.fun == 'log1p':
            loss = 1 - 1 / (self.tau + torch.log1p(gwd_dis))
        else:
            scale = 2 * (_t_det_sqrt.sqrt().sqrt()).clamp(1e-7)
            loss = torch.log1p(torch.sqrt(gwd_dis) / scale)
        return loss


@TASK_UTILS.register_module()
class ClassificationCost(BaseMatchCost):

    def __init__(self, weight=1) -> None:
        super().__init__(weight=weight)

    def __call__(self, data_sample: TextDetDataSample) -> Tensor:

        pred_scores = data_sample.pred_instances.scores
        pred_num = len(pred_scores)
        gt_num = len(data_sample.gt_instances.labels)
        cls_cost = -pred_scores.repeat(gt_num).reshape(pred_num, gt_num).T

        return cls_cost * self.weight
