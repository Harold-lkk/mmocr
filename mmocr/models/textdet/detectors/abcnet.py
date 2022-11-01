# Copyright (c) OpenMMLab. All rights reserved.
from mmocr.registry import MODELS
from .single_stage_text_detector import SingleStageTextDetector


@MODELS.register_module()
class ABCNetDet(SingleStageTextDetector):
    """"""
