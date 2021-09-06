# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch

from mmocr.models.builder import LOSSES
from mmocr.models.textrecog.losses import CTCLoss as _CTCLoss


@LOSSES.register_module(force=True)
class CTCLoss(_CTCLoss):

    def forward(self, outputs, targets_dict, img_metas=None):
        valid_ratios = None
        if img_metas is not None:
            valid_ratios = [
                img_meta.get('valid_ratio', 1.0) for img_meta in img_metas
            ]

        outputs = torch.log_softmax(outputs, dim=2)
        bsz, seq_len = outputs.size(0), outputs.size(1)
        outputs_for_loss = outputs.permute(1, 0, 2).contiguous()  # T * N * C

        if self.flatten:
            targets = targets_dict['flatten_targets']
        else:
            targets = torch.full(
                size=(bsz, seq_len), fill_value=self.blank, dtype=torch.long)
            for idx, tensor in enumerate(targets_dict['targets']):
                valid_len = min(tensor.size(0), seq_len)
                targets[idx, :valid_len] = tensor[:valid_len]

        target_lengths = targets_dict['target_lengths']
        target_lengths = torch.clamp(target_lengths, min=1, max=seq_len).long()

        input_lengths = torch.full(
            size=(bsz, ), fill_value=seq_len, dtype=torch.long)
        if not self.flatten and valid_ratios is not None:
            input_lengths = [
                math.ceil(valid_ratio * seq_len)
                for valid_ratio in valid_ratios
            ]
            input_lengths = torch.Tensor(input_lengths).long()

        loss_ctc = self.ctc_loss(outputs_for_loss,
                                 targets[:sum(target_lengths)], input_lengths,
                                 target_lengths)

        losses = dict(loss_ctc=loss_ctc)

        return losses
