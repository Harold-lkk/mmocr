# Copyright (c) OpenMMLab. All rights reserved.
import torch

ck = torch.load('checkpoints/abcnet_det/abcnet_det.pth')
torch.save(dict(state_dict=ck), 'checkpoints/abcnet_det/abcnet_det.pth')
