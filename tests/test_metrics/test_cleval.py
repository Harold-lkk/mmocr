#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
from mmcv import Config

from mmocr.datasets import build_dataset


def main(cfg_path, output_path):
    cfg = Config.fromfile(cfg_path)
    # build the dataloader
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))

    eval_kwargs = cfg.get('evaluation', {}).copy()
    # hard-code way to remove EvalHook args
    for key in [
            'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best', 'rule'
    ]:
        eval_kwargs.pop(key, None)
    eval_kwargs.update(dict(metric='cleval'))
    outputs = mmcv.load(output_path)
    print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == '__main__':
    cfg_path = 'configs/textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py'
    output_path = 'demo/test/result.pkl'
    main(cfg_path, output_path)
