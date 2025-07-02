from typing import Callable, List, Tuple, Dict
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def sum_tensor(inp: torch.Tensor, axes: int | List[int], keepdim: bool = False) -> torch.Tensor:
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


def get_tp_fp_fn(net_output: torch.Tensor, gt: torch.Tensor, axes: int | Tuple[int, ...] | None = None, mask: torch.Tensor | None = None, square: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))
    shp_x = net_output.shape
    shp_y = gt.shape
    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))
        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)
    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
    tp = sum_tensor(tp, axes, keepdim=False)
    fp = sum_tensor(fp, axes, keepdim=False)
    fn = sum_tensor(fn, axes, keepdim=False)
    return tp, fp, fn


def softmax_helper(x: torch.Tensor) -> torch.Tensor:
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)

def flatten(tensor: torch.Tensor) -> torch.Tensor:
    C = tensor.size(1)
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    transposed = tensor.permute(axis_order).contiguous()
    return transposed.view(C, -1)


class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable | None = softmax_helper, batch_dice: bool = True, do_bg: bool = False, smooth: float = 1.0, square: bool = True) -> None:
        super().__init__()
        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x: torch.Tensor, y: torch.Tensor, loss_mask: torch.Tensor | None = None) -> torch.Tensor:
        shp_x = x.shape
        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
        tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, self.square)
        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()
        return -dc


class SoftDiceLoss_v2(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        targets = F.one_hot(targets, num_classes=probs.size(1)).permute(0, 3, 1, 2).float()
        intersection = torch.sum(probs * targets, dim=(0, 2, 3))
        union = torch.sum(probs + targets, dim=(0, 2, 3))
        dl = 1 - (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = torch.mean(dl)
        return dice_loss


class SSLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable | None = softmax_helper, batch_dice: bool = True, do_bg: bool = False, smooth: float = 1., square: bool = True) -> None:
        super().__init__()
        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.r = 0.1

    def forward(self, net_output: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        shp_x = net_output.shape
        shp_y = gt.shape
        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                gt = gt.view((shp_y[0], 1, *shp_y[1:]))
            if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(shp_x)
                if net_output.device.type == "cuda":
                    y_onehot = y_onehot.cuda(net_output.device.index)
                y_onehot.scatter_(1, gt, 1)
        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))
        if self.apply_nonlin is not None:
            net_output = self.apply_nonlin(net_output)
        bg_onehot = 1 - y_onehot
        squared_error = (y_onehot - net_output)**2
        specificity_part = sum_tensor(squared_error*y_onehot, axes)/(sum_tensor(y_onehot, axes)+self.smooth)
        sensitivity_part = sum_tensor(squared_error*bg_onehot, axes)/(sum_tensor(bg_onehot, axes)+self.smooth)
        ss = self.r * specificity_part + (1-self.r) * sensitivity_part
        if not self.do_bg:
            if self.batch_dice:
                ss = ss[1:]
            else:
                ss = ss[:, 1:]
        ss = ss.mean()
        return ss


class SSLoss_v2(nn.Module):
    def __init__(self, alpha: float = 0.5) -> None:
        super().__init__()
        self.alpha = alpha
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        targets = F.one_hot(targets, num_classes=probs.size(1)).permute(0, 3, 1, 2).float()
        intersection = torch.sum(probs * targets, dim=(0, 2, 3))
        cardinality = torch.sum(probs + targets, dim=(0, 2, 3))
        dice_loss = 1 - (2.0 * intersection + 1e-6) / (cardinality + 1e-6)
        ce_loss = F.cross_entropy(probs, targets, reduction='mean')
        loss = 0.5 * dice_loss.mean() + (1 - 0.5) * ce_loss
        return loss


class IoULoss(nn.Module):
    def __init__(self, apply_nonlin: Callable | None = softmax_helper, batch_dice: bool = True, do_bg: bool = False, smooth: float = 1., square: bool = True) -> None:
        super().__init__()
        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x: torch.Tensor, y: torch.Tensor, loss_mask: torch.Tensor | None = None) -> torch.Tensor:
        shp_x = x.shape
        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
        tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, self.square)
        iou = (tp + self.smooth) / (tp + fp + fn + self.smooth)
        if not self.do_bg:
            if self.batch_dice:
                iou = iou[1:]
            else:
                iou = iou[:, 1:]
        iou = iou.mean()
        return -iou


class IoULoss_v2(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        targets = F.one_hot(targets, num_classes=probs.size(1)).permute(0, 3, 1, 2).float()
        intersection = torch.sum(probs * targets, dim=(0, 2, 3))
        union = torch.sum(probs + targets, dim=(0, 2, 3)) - intersection
        iou = 1 - (intersection + self.smooth) / (union + self.smooth)
        iou_loss = torch.mean(iou)
        return iou_loss


class TverskyLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable | None = softmax_helper, batch_dice: bool = True, do_bg: bool = False, smooth: float = 1., square: bool = True) -> None:
        super().__init__()
        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.alpha = 0.3
        self.beta = 0.7

    def forward(self, x: torch.Tensor, y: torch.Tensor, loss_mask: torch.Tensor | None = None) -> torch.Tensor:
        shp_x = x.shape
        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
        tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, self.square)
        tversky = (tp + self.smooth) / (tp + self.alpha*fp + self.beta*fn + self.smooth)
        if not self.do_bg:
            if self.batch_dice:
                tversky = tversky[1:]
            else:
                tversky = tversky[:, 1:]
        tversky = tversky.mean()
        return -tversky


class TverskyLoss_v2(nn.Module):
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        targets = F.one_hot(targets, num_classes=probs.size(1)).permute(0, 3, 1, 2).float()
        tp = torch.sum(probs * targets, dim=(0, 2, 3))
        fp = torch.sum((1 - targets) * probs, dim=(0, 2, 3))
        fn = torch.sum(targets * (1 - probs), dim=(0, 2, 3))
        tversky = 1 - (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        tversky_loss = torch.mean(tversky)
        return tversky_loss


class FocalTversky_loss(nn.Module):
    def __init__(self, tversky_kwargs: Dict, gamma: float = 0.75) -> None:
        super().__init__()
        self.gamma = gamma
        self.tversky = TverskyLoss(**tversky_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        tversky_loss = 1 + self.tversky(net_output, target)
        focal_tversky = torch.pow(tversky_loss, self.gamma)
        return focal_tversky


class FocalTversky_loss_v2(nn.Module):
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, gamma: float = 1.5, smooth: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        targets = F.one_hot(targets, num_classes=probs.size(1)).permute(0, 3, 1, 2).float()
        tp = torch.sum(probs * targets, dim=(0, 2, 3))
        fp = torch.sum((1 - targets) * probs, dim=(0, 2, 3))
        fn = torch.sum(targets * (1 - probs), dim=(0, 2, 3))
        focal_tversky = (1 - (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)) ** self.gamma
        focal_tversky_loss = torch.mean(focal_tversky)
        return focal_tversky_loss


class AsymLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable | None = softmax_helper, batch_dice: bool = True, do_bg: bool = False, smooth: float = 1., square: bool = True) -> None:
        super().__init__()
        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.beta = 1.5

    def forward(self, x: torch.Tensor, y: torch.Tensor, loss_mask: torch.Tensor | None = None) -> torch.Tensor:
        shp_x = x.shape
        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
        tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, self.square)
        weight = (self.beta**2)/(1+self.beta**2)
        asym = (tp + self.smooth) / (tp + weight*fn + (1-weight)*fp + self.smooth)
        if not self.do_bg:
            if self.batch_dice:
                asym = asym[1:]
            else:
                asym = asym[:, 1:]
        asym = asym.mean()
        return -asym


class AsymLoss_v2(nn.Module):
    def __init__(self, alpha: float = 0.5, gamma: float = 2.0, smooth: float = 1e-5) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=probs.size(1)).permute(0, 3, 1, 2).float()
        pos_loss = -self.alpha * (1 - probs) ** self.gamma * targets_one_hot * torch.log(probs + self.smooth)
        neg_loss = -(1 - self.alpha) * probs ** self.gamma * (1 - targets_one_hot) * torch.log(1 - probs + self.smooth)
        loss = pos_loss + neg_loss
        return loss.mean()


class ExpLog_loss(nn.Module):
    def __init__(self, soft_dice_kwargs: Dict, wce_kwargs: Dict, gamma: float = 0.3) -> None:
        super().__init__()
        self.wce = WeightedCrossEntropyLoss(**wce_kwargs)
        self.dc = SoftDiceLoss_v2(**soft_dice_kwargs)
        self.gamma = gamma

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dc_loss = -self.dc(net_output, target)
        wce_loss = self.wce(net_output, target)
        explog_loss = 0.8*torch.pow(-torch.log(torch.clamp(dc_loss, 1e-6)), self.gamma) + 0.2*wce_loss
        return explog_loss


class FocalLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable | None = softmax_helper, alpha: float | List[float] | np.ndarray | None = None, gamma: int = 2, balance_index: int = 0, smooth: float = 1e-4, size_average: bool = True) -> None:
        super().__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average
        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError("smooth value should be in [0,1]")

    def forward(self, logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]
        if logit.dim() > 2:
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        alpha = self.alpha
        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha
        else:
            raise TypeError("Not support alpha type")
        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)
        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)
        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth/(num_class-1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()
        gamma = self.gamma
        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


class LovaszSoftmax(nn.Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def prob_flatten(self, input: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert input.dim() in [4, 5]
        num_class = input.size(1)
        if input.dim() == 4:
            input = input.permute(0, 2, 3, 1).contiguous()
            input_flatten = input.view(-1, num_class)
        elif input.dim() == 5:
            input = input.permute(0, 2, 3, 4, 1).contiguous()
            input_flatten = input.view(-1, num_class)
        target_flatten = target.view(-1)
        return input_flatten, target_flatten

    def lovasz_softmax_flat(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = inputs.size(1)
        losses = []
        for c in range(num_classes):
            target_c = (targets == c).float()
            if num_classes == 1:
                input_c = inputs[:, 0]
            else:
                input_c = inputs[:, c]
            loss_c = (torch.autograd.Variable(target_c) - input_c).abs()
            loss_c_sorted, loss_index = torch.sort(loss_c, 0, descending=True)
            target_c_sorted = target_c[loss_index]
            losses.append(torch.dot(loss_c_sorted, torch.autograd.Variable(lovasz_grad(target_c_sorted))))
        losses = torch.stack(losses)
        if self.reduction == "none":
            loss = losses
        elif self.reduction == "sum":
            loss = losses.sum()
        else:
            loss = losses.mean()
        return loss

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs, targets = self.prob_flatten(inputs, targets)
        losses = self.lovasz_softmax_flat(inputs, targets)
        return losses
    

class TopKLoss(nn.Module):
    def __init__(self, weight: torch.Tensor | None = None, ignore_index: int = -100, k: int = 10) -> None:
        super().__init__()
        self.k = k
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction="none")

    def forward(self, inp: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pixel_losses = self.cross_entropy(inp, target)
        pixel_losses = pixel_losses.view(-1)
        num_voxels = pixel_losses.numel()
        res, _ = torch.topk(pixel_losses, int(num_voxels * self.k / 100), sorted=False)
        return res.mean()


class WeightedCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    def __init__(self, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, inp: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.long()
        num_classes = inp.size()[1]
        i0 = 1
        i1 = 2
        while i1 < len(inp.shape):
            inp = inp.transpose(i0, i1)
            i0 += 1
            i1 += 1
        inp = inp.contiguous()
        inp = inp.view(-1, num_classes)
        target = target.view(-1,)
        wce_loss = torch.nn.CrossEntropyLoss(weight=self.weight)
        return wce_loss(inp, target)