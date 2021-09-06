import torch
import torch.nn.functional as F
from torch import nn

from mmocr.models.builder import LOSSES, build_loss


@LOSSES.register_module()
class MaskL1Loss(nn.Module):

    def __init__(self, beta=1.0, eps=1e-6, loss_weight=1):
        super().__init__()
        self.beta = beta
        self.eps = eps
        self.loss_weight = loss_weight

    def forward(self, pred, gt, mask):
        loss = torch.abs((pred - gt) * mask).sum() / (mask.sum() + self.eps)
        return loss * self.loss_weight


@LOSSES.register_module()
class MaskSmoothL1Loss(nn.Module):

    def __init__(self, beta=1.0, eps=1e-6, loss_weight=1):
        super().__init__()
        self.beta = beta
        self.eps = eps
        self.loss_weight = loss_weight

    def forward(self, pred, gt, mask):
        assert pred.size() == gt.size() and gt.numel() > 0
        diff = torch.abs((pred - gt) * mask)
        loss = torch.where(diff < self.beta, 0.5 * diff * diff / self.beta,
                           diff - 0.5 * self.beta).sum() / (
                               mask.sum() + self.eps)
        return loss * self.loss_weight


@LOSSES.register_module()
class MaskBalanceBCELoss(nn.Module):

    def __init__(self,
                 reduction='mean',
                 negative_ratio=3,
                 eps=1e-6,
                 loss_weight=1):
        super().__init__()
        assert isinstance(eps, float)
        self.eps = eps
        self.negative_ratio = negative_ratio
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, gt, mask):
        positive = (gt * mask)
        negative = ((1 - gt) * mask)
        positive_count = int(positive.float().sum())
        negative_count = min(
            int(negative.float().sum()),
            int(positive_count * self.negative_ratio))

        assert gt.max() <= 1 and gt.min() >= 0
        assert pred.max() <= 1 and pred.min() >= 0
        loss = F.binary_cross_entropy(pred, gt, reduction=self.reduction)
        positive_loss = loss * positive.float()
        negative_loss = loss * negative.float()

        negative_loss, _ = torch.topk(negative_loss.view(-1), negative_count)

        balance_loss = (positive_loss.sum() + negative_loss.sum()) / (
            positive_count + negative_count + self.eps)

        return balance_loss * self.loss_weight


@LOSSES.register_module()
class MaskDiceLoss(nn.Module):

    def __init__(self, eps=1e-6, loss_weight=1):
        super().__init__()
        assert isinstance(eps, float)
        self.eps = eps
        self.loss_weight = loss_weight

    def forward(self, pred, target, mask=None):

        pred = pred.contiguous().view(pred.size()[0], -1)
        target = target.contiguous().view(target.size()[0], -1)

        if mask is not None:
            mask = mask.contiguous().view(mask.size()[0], -1)
            pred = pred * mask
            target = target * mask

        a = torch.sum(pred * target)
        b = torch.sum(pred)
        c = torch.sum(target)
        d = (2 * a) / (b + c + self.eps)

        return (1 - d) * self.loss_weight


@LOSSES.register_module()
class SpeDBLoss(nn.Module):
    """The class for implementing DBNet loss.

    This is partially adapted from https://github.com/MhLiao/DB.
    """

    def __init__(self, probability_head, threshold_head, diff_binary_head):
        """Initialization.

        Args:
            alpha (float): The binary loss coef.
            beta (float): The threshold loss coef.
            reduction (str): The way to reduce the loss.
            negative_ratio (float): The ratio of positives to negatives.
            eps (float): Epsilon in the threshold loss function.
            bbce_loss (bool): Whether to use balanced bce for probability loss.
                If False, dice loss will be used instead.
        """
        super().__init__()
        self.probability_loss = build_loss(probability_head)
        self.threshold_loss = build_loss(threshold_head)
        self.diff_binary_loss = build_loss(diff_binary_head)

    def forward(self, preds, downsample_ratio, gt_shrink, gt_shrink_mask,
                gt_thr, gt_thr_mask):
        """Compute DBNet loss.

        Args:
            preds (tensor): The output tensor with size of Nx3xHxW.
            downsample_ratio (float): The downsample ratio for the
                ground truths.
            gt_shrink (list[BitmapMasks]): The mask list with each element
                being the shrinked text mask for one img.
            gt_shrink_mask (list[BitmapMasks]): The effective mask list with
                each element being the shrinked effective mask for one img.
            gt_thr (list[BitmapMasks]): The mask list with each element
                being the threshold text mask for one img.
            gt_thr_mask (list[BitmapMasks]): The effective mask list with
                each element being the threshold effective mask for one img.

        Returns:
            results(dict): The dict for dbnet losses with loss_prob,
                loss_db and loss_thresh.
        """
        assert isinstance(downsample_ratio, float)

        assert isinstance(gt_shrink, list)
        assert isinstance(gt_shrink_mask, list)
        assert isinstance(gt_thr, list)
        assert isinstance(gt_thr_mask, list)

        pred_prob = preds[:, 0, :, :]
        pred_thr = preds[:, 1, :, :]
        pred_db = preds[:, 2, :, :]

        keys = ['gt_shrink', 'gt_shrink_mask', 'gt_thr', 'gt_thr_mask']
        gt = {}
        for k in keys:
            gt[k] = eval(k)
            gt[k] = [item.rescale(downsample_ratio) for item in gt[k]]
            gt[k] = [
                gt[k][batch_inx].to_tensor(torch.float32, preds.device)
                for batch_inx in range(len(gt[k]))
            ]
            gt[k] = torch.cat(gt[k])
        gt['gt_shrink'] = (gt['gt_shrink'] > 0).float()

        probability_loss = self.probability_loss(pred_prob, gt['gt_shrink'],
                                                 gt['gt_shrink_mask'])

        threshold_loss = self.threshold_loss(pred_thr, gt['gt_thr'],
                                             gt['gt_thr_mask'])
        diff_binary_loss = self.diff_binary_loss(pred_db, gt['gt_shrink'],
                                                 gt['gt_shrink_mask'])

        results = dict(
            probability_loss=probability_loss,
            threshold_loss=threshold_loss,
            diff_binary_loss=diff_binary_loss)

        return results
