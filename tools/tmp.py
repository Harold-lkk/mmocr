# Copyright (c) OpenMMLab. All rights reserved.
import pickle

import mmdet.models  # noqa: F401
from mmengine.structures import InstanceData  # noqa: F401

from mmocr.models.textdet import ABCNetDetPostprocessor
from mmocr.structures import TextDetDataSample

postprocessor = ABCNetDetPostprocessor(
    rescale_fields=['polygons', 'bboxes', 'bezier'],
    use_sigmoid_cls=True,
    strides=[8, 16, 32, 64, 128],
    bbox_coder=dict(type='mmdet.DistancePointBBoxCoder'),
    with_bezier=True,
    test_cfg=dict(
        # rescale_fields=['polygon', 'bboxes', 'bezier'],
        nms_pre=1000,
        nms=dict(type='nms', iou_threshold=0.5),
        score_thr=0.3))

pred_results = pickle.load(
    open('/mnt/hdd/project/AdelaiDet/checkpoints/outputs_dict.pickle', 'rb'))

# post_res = pickle.load(
#     open('/mnt/hdd/project/AdelaiDet/checkpoints/results.pickle', 'rb'))

data_sample = TextDetDataSample(
    img_shape=(1344, 1024),
    scale_factor=(1 / 0.359, 1 / 0.35901271503365745),
    origin_shape=(1337, 1000))
pred_results = [
    pred_results['logits_pred'], pred_results['reg_pred'],
    pred_results['ctrness_pred'], pred_results['top_feats']
]
abcnet_det_res = postprocessor(pred_results, [data_sample])
