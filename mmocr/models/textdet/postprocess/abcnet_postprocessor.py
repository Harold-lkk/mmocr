# Copyright (c) OpenMMLab. All rights reserved.
import torch
# from mmcv.runner import force_fp32
from mmcv.ops import batched_nms
from mmdet.core import build_bbox_coder, multi_apply
from mmdet.core.anchor.point_generator import MlvlPointGenerator
from mmdet.core.utils import filter_scores_and_topk, select_single_mlvl

from mmocr.models.builder import POSTPROCESSOR
from mmocr.utils.box_util import bezier_to_polygon
from .base_postprocessor import BaseTextDetPostProcessor


@POSTPROCESSOR.register_module()
class ABCNetTextDetProcessor(BaseTextDetPostProcessor):

    def __init__(self,
                 num_classes=1,
                 use_sigmoid_cls=True,
                 strides=(4, 8, 16, 32, 64),
                 bbox_coder=dict(type='DistancePointBBoxCoder'),
                 text_repr_type='poly',
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):
        super().__init__(
            text_repr_type=text_repr_type,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            **kwargs)
        self.prior_generator = MlvlPointGenerator(strides)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.use_sigmoid_cls = use_sigmoid_cls
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

    # @force_fp32(apply_to=('cls_scores', 'bbox_preds'))

    def filter_and_location(self,
                            det_results,
                            img_meta=None,
                            nms_pre=-1,
                            score_thr=0,
                            max_per_img=100,
                            nms=dict(type='IoULoss', loss_weight=1.0),
                            **kwargs):
        cls_scores = det_results.get('cls_scores')
        bbox_preds = det_results.get('bbox_preds')
        centerness_preds = det_results.get('centerness_preds')
        bezier_preds = det_results.get('bezier_preds')
        mlvl_priors = det_results.get('mlvl_priors')

        parameters = dict(
            img_shape=img_meta['img_shape'],
            nms_pre=nms_pre,
            score_thr=score_thr)
        (mlvl_bboxes, mlvl_scores, mlvl_labels, mlvl_score_factors,
         mlvl_beziers) = multi_apply(self._single_level, cls_scores,
                                     bbox_preds, centerness_preds,
                                     bezier_preds, mlvl_priors, parameters)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_labels = torch.cat(mlvl_labels)
        mlvl_beziers = torch.cat(mlvl_beziers)

        if mlvl_score_factors is not None:
            # TODO： Add sqrt operation in order to be consistent with
            #  the paper.
            mlvl_score_factors = torch.cat(mlvl_score_factors)
            mlvl_scores = mlvl_scores * mlvl_score_factors

        if mlvl_bboxes.numel() == 0:
            det_bboxes = torch.cat([mlvl_bboxes, mlvl_scores[:, None]], -1)
            return det_bboxes, mlvl_labels

        det_bboxes, keep_idxs = batched_nms(mlvl_bboxes, mlvl_scores,
                                            mlvl_labels, nms)
        results = dict(
            bboxes=det_bboxes[:max_per_img],
            labels=mlvl_labels[keep_idxs][:max_per_img],
            bezier=mlvl_beziers[keep_idxs][:max_per_img])

        return results

    def split_results(self, pred_results, img_metas, **kwargs):

        results = []
        cls_scores = pred_results.get('cls_scores')
        bbox_preds = pred_results.get('bbox_preds')
        centerness_preds = pred_results.get('centerness_preds')
        bezier_preds = pred_results.get('bezier_preds')
        num_levels = len(cls_scores)
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device)
        for img_id in range(len(img_metas)):
            single_results = dict(
                cls_scores=select_single_mlvl(cls_scores, img_id),
                bbox_preds=select_single_mlvl(bbox_preds, img_id),
                centerness_preds=select_single_mlvl(centerness_preds, img_id),
                bezier_preds=select_single_mlvl(bezier_preds, img_id),
                mlvl_priors=mlvl_priors)
            results.append(single_results)
        return results

    def reconstruct_text_instance(self, results, **kwargs):
        bezier_points = results['bezier'].reshape(-1, 2, 4, 2)
        results['polygon'] = list(map(bezier_to_polygon, bezier_points))
        return results

    def _single_level(self, score_thr, nms_pre, img_shape, cls_score,
                      bbox_pred, centerness_pred, bezier_pred, priors):
        assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
        bezier_pred = bezier_pred.permute(1, 2, 0).reshape(-1, 8, 2)
        centerness_pred = centerness_pred.permute(1, 2,
                                                  0).reshape(-1).sigmoid()
        cls_score = cls_score.permute(1, 2, 0).reshape(-1,
                                                       self.cls_out_channels)
        if self.use_sigmoid_cls:
            scores = cls_score.sigmoid()
        else:
            # remind that we set FG labels to [0, num_class-1]
            # since mmdet v2.0
            # BG cat_id: num_class
            scores = cls_score.softmax(-1)[:, :-1]

        # After https://github.com/open-mmlab/mmdetection/pull/6268/,
        # this operation keeps fewer bboxes under the same `nms_pre`.
        # There is no difference in performance for most models. If you
        # find a slight drop in performance, you can set a larger
        # `nms_pre` than before.
        results = filter_scores_and_topk(
            scores, score_thr, nms_pre,
            dict(bbox_pred=bbox_pred, priors=priors))
        scores, labels, keep_idxs, filtered_results = results

        bbox_pred = filtered_results['bbox_pred']
        priors = filtered_results['priors']
        centerness_pred = centerness_pred[keep_idxs]
        bezier_pred = bezier_pred[keep_idxs]

        bboxes = self.bbox_coder.decode(priors, bbox_pred, max_shape=img_shape)
        bezier_pred = priors + bezier_pred
        bezier_pred[:0].clamp_(min=0, max=img_shape[1])
        bezier_pred[:1].clamp_(min=0, max=img_shape[0])
        return bboxes, scores, labels, centerness_pred, bezier_pred
