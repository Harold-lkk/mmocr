# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Sequence

from .base import BaseTextDetPostProcessor


class FSGPostProcessor(BaseTextDetPostProcessor):

    def __init__(
        self,
        text_repr_type: str = 'poly',
        rescale_fields: Optional[Sequence[str]] = None,
        train_cfg: Optional[Dict] = None,
        test_cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__(text_repr_type, rescale_fields, train_cfg, test_cfg)
