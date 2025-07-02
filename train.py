from typing import Tuple
from pathlib import Path
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import wandb
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader

from models.unet import UNet
from dataset import SegmentationDataset, collate_fn
from utils import get_transform, mask_transform, EMA
from get_loss import get_composite_criterion
from models.vit import ViTSegmentation
from models.dino import DINOSegmentationModel


color_map = cm.get_cmap('tab20', 18)
fixed_colors = np.array([color_map(i)[:3] for i in range(18)]) * 255


def mask_to_color(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_idx in range(18):
        color_mask[mask == class_idx] = fixed_colors[class_idx]
    return color_mask


def create_combined_image(
    x: torch.Tensor,
    y: torch.Tensor,
    y_pred: torch.Tensor,
    mean: list[float] = [0.485, 0.456, 0.406],
    std: list[float] = [0.229, 0.224, 0.225]
) -> np.ndarray:
    batch_size, _, height, width = x.shape
    combined_height = height * 3
    combined_width = width * batch_size
    combined_image = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)

    for i in range(batch_size):
        image = x[i].cpu().permute(1, 2, 0).numpy()
        image = (image * std + mean).clip(0, 1)
        image = (image * 255).astype(np.uint8)
        true_mask = y[i].cpu().numpy()
        true_mask_color = mask_to_color(true_mask)
        pred_mask = y_pred[i].cpu().numpy()
        pred_mask_color = mask_to_color(pred_mask)
        combined_image[:height, i * width:(i + 1) * width, :] = image
        combined_image[height:2 * height, i * width:(i + 1) * width, :] = true_mask_color
        combined_image[2 * height:, i * width:(i + 1) * width, :] = pred_mask_color
    return combined_image


def compute_metrics(y_pred: torch.Tensor, y: torch.Tensor, num_classes: int = 18) -> Tuple[float, float, float, float, float, float]:
    pred_mask = y_pred.unsqueeze(-1) == torch.arange(num_classes, device=y_pred.device).reshape(1, 1, 1, -1)
    target_mask = y.unsqueeze(-1) == torch.arange(num_classes, device=y.device).reshape(1, 1, 1, -1)
    class_present = (target_mask.sum(dim=(0, 1, 2)) > 0).float()
    tp = (pred_mask & target_mask).sum(dim=(0, 1, 2)).float()
    fp = (pred_mask & ~target_mask).sum(dim=(0, 1, 2)).float()
    fn = (~pred_mask & target_mask).sum(dim=(0, 1, 2)).float()
    tn = (~pred_mask & ~target_mask).sum(dim=(0, 1, 2)).float()
    overall_tp = tp.sum()
    overall_fp = fp.sum()
    overall_fn = fn.sum()
    overall_tn = tn.sum()
    precision = tp / (tp + fp).clamp(min=1e-8)
    recall = tp / (tp + fn).clamp(min=1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    macro_precision = ((precision * class_present).sum() / class_present.sum().clamp(min=1e-8)).item()
    macro_recall = ((recall * class_present).sum() / class_present.sum().clamp(min=1e-8)).item()
    macro_accuracy = accuracy.mean().item()
    micro_precision = (overall_tp / (overall_tp + overall_fp).clamp(min=1e-8)).item()
    micro_recall = (overall_tp / (overall_tp + overall_fn).clamp(min=1e-8)).item()
    global_accuracy = ((y_pred == y).sum() / (y.shape[0] * y.shape[1] * y.shape[2])).item()
    return macro_precision, macro_recall, macro_accuracy, micro_precision, micro_recall, global_accuracy


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model on human parsing dataset")
    parser.add_argument("--data-path", type=str, default="mattmdjaga/human_parsing_dataset", help="Path to the data")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training and testing")
    parser.add_argument("--pin-memory", type=bool, default=True, help="Pin memory for DataLoader")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of workers for DataLoader")
    parser.add_argument("--num-epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimizer type")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate for the optimizer")
    parser.add_argument("--max-norm", type=float, default=1.0, help="Maximum gradient norm for clipping")
    parser.add_argument("--logs-dir", type=str, default="dino-logs", help="Directory for saving logs")
    parser.add_argument("--model", type=str, default="dino", choices=["unet", "vit", "dino"], help="Model class name")
    parser.add_argument("--losses-path", type=str, default="losses_config.json", help="Path to the losses")
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["fp16", "bf16", "fp8", "no"], help="Value of the mixed precision")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2, help="Value of the gradient accumulation steps")
    parser.add_argument("--project-name", type=str, default="human_parsing_segmentation_ttk", help="WandB project name")
    parser.add_argument("--save-frequency", type=int, default=4, help="Frequency of saving model weights")
    parser.add_argument("--log-steps", type=int, default=400, help="Number of steps between logging training images and metrics")
    parser.add_argument("--seed", type=int, default=42, help="Value of the seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    set_seed(args.seed)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, mixed_precision=args.mixed_precision)

    with open(args.losses_path, "r") as fp:
        losses_config = json.load(fp)

    with accelerator.main_process_first():
        logs_dir = Path(args.logs_dir)
        logs_dir.mkdir(exist_ok=True)
        wandb.init(project=args.project_name, dir=logs_dir)
        wandb.save(args.losses_path)
    
    optimizer_class = getattr(torch.optim, args.optimizer)
    
    if args.model == "unet":
        model = UNet().to(accelerator.device)
        optimizer = optimizer_class(model.parameters(), lr=args.learning_rate)
    elif args.model == "vit":
        model = ViTSegmentation().to(accelerator.device)
        optimizer = optimizer_class(model.parameters(), lr=args.learning_rate)
    elif args.model == "dino":
        model = DINOSegmentationModel().to(accelerator.device)
        optimizer = optimizer_class(model.segmentation_head.parameters(), lr=args.learning_rate)
    else:
        raise NotImplementedError("Incorrect model name")

    transform = get_transform(model.mean, model.std)

    dataset = load_dataset(args.data_path, split="train")
    train_dataset = SegmentationDataset(dataset, train=True, transform=transform, target_transform=mask_transform)
    valid_dataset = SegmentationDataset(dataset, train=False, transform=transform, target_transform=mask_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory, collate_fn=collate_fn)

    criterion = get_composite_criterion(losses_config)
    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs * len(train_loader))

    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(model, optimizer, train_loader, lr_scheduler)

    best_accuracy = 0

    print(f"params: {sum([p.numel() for p in model.parameters()])/1e6:.2f} M")
    print(f"trainable params: {sum([p.numel() for p in model.parameters() if p.requires_grad])/1e6:.2f} M")

    train_loss_ema, train_macro_precision_ema, train_macro_recall_ema, train_macro_accuracy_ema, train_micro_precision_ema, train_micro_recall_ema, train_global_accuracy_ema = EMA(), EMA(), EMA(), EMA(), EMA(), EMA(), EMA()
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        print(f"trainable params: {sum([p.numel() for p in model.parameters() if p.requires_grad])/1e6:.2f} M")
        exit()
        pbar = tqdm(train_loader, desc=f"Train epoch {epoch}/{args.num_epochs}")
        for index, (x, y) in enumerate(pbar):
            x, y = x.to(accelerator.device), y.squeeze(1).to(accelerator.device)
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    output = model(x)
                    loss = criterion(output, y)
                    accelerator.backward(loss)
                    train_loss = loss.item()
                    grad_norm = None
                    _, y_pred = output.max(dim=1)
                    train_macro_precision, train_macro_recall, train_macro_accuracy, train_micro_precision, train_micro_recall, train_global_accuracy = compute_metrics(y_pred, y)
                    if accelerator.sync_gradients:
                        grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_norm).item()
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                    if (index + 1) % args.log_steps == 0 and accelerator.is_main_process:
                        images_to_log = []
                        combined_image = create_combined_image(x, y, y_pred)
                        images_to_log.append(wandb.Image(combined_image, caption=f"Combined Image (Train, Epoch {epoch}, Batch {index})"))
                        wandb.log({"train_samples": images_to_log})
                    pbar.set_postfix({"loss": train_loss_ema(train_loss), "macro_precision": train_macro_precision_ema(train_macro_precision), "macro_recall": train_macro_recall_ema(train_macro_recall), "macro_accuracy": train_macro_accuracy_ema(train_macro_accuracy), "micro_precision": train_micro_precision_ema(train_micro_precision), "micro_recall": train_micro_recall_ema(train_micro_recall), "global_accuracy": train_global_accuracy_ema(train_global_accuracy)})
                    log_data = {
                        "train/epoch": epoch,
                        "train/loss": train_loss,
                        "train/macro_accuracy": train_macro_accuracy,
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                        "train/macro_precision": train_macro_precision,
                        "train/macro_recall": train_macro_recall,
                        "train/micro_precision": train_micro_precision,
                        "train/micro_recall": train_micro_recall,
                        "train/global_accuracy": train_global_accuracy,
                    }
                    if grad_norm is not None:
                        log_data["train/grad_norm"] = grad_norm
                    if accelerator.is_main_process:
                        wandb.log(log_data)
        accelerator.wait_for_everyone()
        
        model.eval()
        valid_loss, valid_macro_accuracies, valid_macro_precisions, valid_macro_recalls, valid_global_accuracies, valid_micro_precisions, valid_micro_recalls = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        with torch.inference_mode():
            pbar = tqdm(valid_loader, desc=f"Val epoch {epoch}/{args.num_epochs}")
            for index, (x, y) in enumerate(valid_loader):
                x, y = x.to(accelerator.device), y.squeeze(1).to(accelerator.device)
                output = model(x)
                _, y_pred = output.max(dim=1)
                if (index + 1) % args.log_steps == 0 and accelerator.is_main_process:
                    images_to_log = []
                    combined_image = create_combined_image(x, y, y_pred)
                    images_to_log.append(wandb.Image(combined_image, caption=f"Combined Image (Validation, Epoch {epoch})"))
                    wandb.log({"valid_samples": images_to_log})
                valid_macro_precision, valid_macro_recall, valid_macro_accuracy, valid_micro_precision, valid_micro_recall, valid_global_accuracy = compute_metrics(y_pred, y)
                valid_macro_precisions += valid_macro_precision
                valid_macro_recalls += valid_macro_recall
                valid_macro_accuracies += valid_macro_accuracy
                valid_micro_precisions += valid_micro_precision
                valid_micro_recalls += valid_micro_recall
                valid_global_accuracies += valid_global_accuracy
                loss = criterion(output, y)
                valid_loss += loss.item()
        valid_loss = valid_loss / len(valid_loader)
        valid_macro_accuracies = valid_macro_accuracies / len(valid_loader)
        valid_macro_precisions = valid_macro_precisions / len(valid_loader)
        valid_macro_recalls = valid_macro_recalls / len(valid_loader)
        valid_global_accuracies = valid_global_accuracies / len(valid_loader)
        valid_micro_precisions = valid_micro_precisions / len(valid_loader)
        valid_micro_recalls = valid_micro_recalls / len(valid_loader)
        accelerator.print(f"loss: {valid_loss:.3f}, valid_macro_precision: {valid_macro_precisions:.3f}, valid_macro_recall: {valid_macro_recalls:.3f}, valid_macro_accuracy: {valid_macro_accuracies:.3f}, valid_micro_precision: {valid_micro_precisions:.3f}, valid_micro_recall: {valid_micro_recalls:.3f}, valid_global_accuracy: {valid_global_accuracies:.3f}")
        if accelerator.is_main_process:
            wandb.log(
                {
                    "val/epoch": epoch,
                    "val/loss": valid_loss,
                    "val/macro_accuracy": valid_macro_accuracies,
                    "val/macro_precision": valid_macro_precisions,
                    "val/macro_recall": valid_macro_recalls,
                    "val/global_accuracy": valid_global_accuracies,
                    "val/micro_precision": valid_micro_precisions,
                    "val/micro_recall": valid_micro_recalls,
                }
            )
            if valid_global_accuracies > best_accuracy:
                best_accuracy = valid_global_accuracies
                if args.model in ["dino", "vit"]:
                    accelerator.save(model.segmentation_head.state_dict(), logs_dir / f"checkpoint-best.pth")
                else:
                    accelerator.save(model.state_dict(), logs_dir / f"checkpoint-best.pth")
                accelerator.print(f"new best_accuracy {best_accuracy}, {epoch=}")
            if epoch % args.save_frequency == 0:
                if args.model in ["dino", "vit"]:
                    accelerator.save(model.segmentation_head.state_dict(), logs_dir / f"checkpoint-{epoch:09}.pth")
                else:
                    accelerator.save(model.state_dict(), logs_dir / f"checkpoint-{epoch:09}.pth")
        accelerator.wait_for_everyone()

    accelerator.wait_for_everyone()
    wandb.finish()


if __name__ == "__main__":
    main()