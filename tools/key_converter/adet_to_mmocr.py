# Copyright (c) OpenMMLab. All rights reserved.
import torch

abcnet_det = torch.load('checkpoints/abcnet/abcnet_det.pth')
abcnet_rec = torch.load('checkpoints/abcnet/abcnet_rec.pth')
abcnet = dict(state_dict=dict())
abcnet['state_dict'].update(abcnet_det['state_dict'])
for k, v in abcnet_rec['state_dict'].items():

    abcnet['state_dict']['roi_head.rec_head.' + k] = v

torch.save(abcnet, 'checkpoints/abcnet/abcnet.pth')

output_txt = 'tools/key_converter/abcnet_merge.txt'
with open(output_txt, 'w') as f:
    for k in abcnet['state_dict'].keys():
        # for k in model['model'].keys():
        f.write(k + '\n')
