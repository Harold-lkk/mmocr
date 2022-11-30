# Copyright (c) OpenMMLab. All rights reserved.
# from mmengine import load_checkpoint, save_checkpoint
import mmdet.models  # noqa: F401
import torch  # noqa: F401
from mmengine import Config
from mmengine.runner import load_checkpoint

from mmocr.registry import MODELS
from mmocr.structures import TextSpottingDataSample
from mmocr.utils.setup_env import register_all_modules

config_path = 'configs/textspotting/abcnet/_base_abcnet-det_resnet50_fpn.py'

config = Config.fromfile(config_path)
register_all_modules(True)

model = MODELS.build(config.model)
model.eval()
# output_txt = 'tools/key_converter/abcnet.txt'
# with open(output_txt, 'w') as f:
#     for k in model.state_dict().keys():
#         #for k in model['model'].keys():
#         f.write(k + '\n')
# output_path = 'tools/key_converter/abcnet.pth'
inputs = torch.load('checkpoints/abcnet/debug/input.pth').cuda()
checkpoint = load_checkpoint(
    model, 'checkpoints/abcnet/abcnet.pth', map_location='cpu')
model.cuda()

data_sample = TextSpottingDataSample(
    img_shape=(576, 1024), scale_factor=(1 / 0.359, 1 / 0.35901271503365745))
output = model.predict(inputs, [data_sample])
