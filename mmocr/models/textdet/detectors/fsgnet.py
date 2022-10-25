# Copyright (c) OpenMMLab. All rights reserved.
from mmocr.registry import MODELS
from .single_stage_text_detector import SingleStageTextDetector


@MODELS.register_module()
class FSGNet(SingleStageTextDetector):
    """The class for implementing FSGNet text detector:

    Few Could Be Better Than All: Feature Sampling and Grouping for Scene Text
    Detection [https://arxiv.org/abs/2203.15221v2].
    """
