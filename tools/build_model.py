# Copyright (c) OpenMMLab. All rights reserved.
# from mmengine import load_checkpoint, save_checkpoint
import mmdet.models  # noqa: F401
import torch  # noqa: F401
from mmengine import Config
from mmengine.runner import load_checkpoint

from mmocr.registry import MODELS
from mmocr.utils.setup_env import register_all_modules

config_path = 'configs/textdet/abcnet/_base_abcnet-det_resnet50_fpn.py'

config = Config.fromfile(config_path)
register_all_modules(True)

model = MODELS.build(config.model)
model.eval()
output_path = 'tools/key_converter/abcnet_det.pth'
# with open(output_path, 'w') as f:
#     for k in model.state_dict().keys():
#         #for k in model['model'].keys():
#         f.write(k + '\n')

# torch.save(model.state_dict(), output_path)
checkpoint = load_checkpoint(
    model, 'checkpoints/abcnet_det/abcnet_det.pth', map_location='cpu')
