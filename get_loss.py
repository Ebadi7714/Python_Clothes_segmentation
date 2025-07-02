from typing import Dict, Callable
import torch.nn as nn
import torch

from losses import SoftDiceLoss, SSLoss, IoULoss, TverskyLoss, FocalTversky_loss, AsymLoss, ExpLog_loss, FocalLoss, LovaszSoftmax, TopKLoss, WeightedCrossEntropyLoss, SoftDiceLoss_v2, IoULoss_v2, TverskyLoss_v2, FocalTversky_loss_v2, AsymLoss_v2, SSLoss_v2


def get_loss(loss_type: str) -> Callable | None:
    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif loss_type == "SoftDiceLoss":
        return SoftDiceLoss()
    elif loss_type == "SSLoss":
        return SSLoss()
    elif loss_type == "IoULoss":
        return IoULoss()
    elif loss_type == "TverskyLoss":
        return TverskyLoss()
    elif loss_type == "FocalTversky_loss":
        tversky_kwargs = {
            "apply_nonlin": None,
            "batch_dice": False,
            "do_bg": True,
            "smooth": 1.0,
            "square": False
        }
        return FocalTversky_loss(tversky_kwargs=tversky_kwargs)
    elif loss_type == "AsymLoss":
        return AsymLoss()
    elif loss_type == "ExpLog_loss":
        soft_dice_kwargs = {
            "smooth": 1.0
        }
        wce_kwargs = {
            "weight": None
        }
        return ExpLog_loss(soft_dice_kwargs=soft_dice_kwargs, wce_kwargs=wce_kwargs)
    elif loss_type == "FocalLoss":
        return FocalLoss()
    elif loss_type == "LovaszSoftmax":
        return LovaszSoftmax()
    elif loss_type == "TopKLoss":
        return TopKLoss()
    elif loss_type == "WeightedCrossEntropyLoss":
        return WeightedCrossEntropyLoss()
    elif loss_type == "SoftDiceLoss_v2":
        return SoftDiceLoss_v2()
    elif loss_type == "IoULoss_v2":
        return IoULoss_v2()
    elif loss_type == "TverskyLoss_v2":
        return TverskyLoss_v2()
    elif loss_type == "FocalTversky_loss_v2":
        return FocalTversky_loss_v2()
    elif loss_type == "AsymLoss_v2":
        return AsymLoss_v2()
    elif loss_type == "SSLoss_v2":
        return SSLoss_v2()
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")


def get_composite_criterion(losses_config: Dict[str, float]) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    losses = []
    weights = []

    for loss_name, weight in losses_config.items():
        if weight != 0.0:
            loss_fn = get_loss(loss_name)
            if loss_fn is not None:
                losses.append(loss_fn)
                weights.append(weight)

    def composite_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        for loss_fn, weight in zip(losses, weights):
            total_loss += weight * loss_fn(output, target)
        return total_loss

    return composite_loss