import torch
import torch.nn as nn
from transformers import ViTModel

from src.models.segmentation_head import SegmentationHead


class ViTSegmentation(nn.Module):
    def __init__(self, image_size: int = 224, num_classes: int = 18) -> None:
        super().__init__()
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]
        self.backbone = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.segmentation_head = SegmentationHead(in_channels=768, num_classes=num_classes)
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.size()
        assert height == width == self.backbone.config.image_size, "The image must match the size required by the ViT model"
        outputs = self.backbone(pixel_values=x).last_hidden_state
        patch_dim = int(height / self.backbone.config.patch_size)
        outputs = outputs[:, 1:, :]
        outputs = outputs.permute(0, 2, 1).view(batch_size, -1, patch_dim, patch_dim)
        masks = self.segmentation_head(outputs)
        return masks


def main() -> None:
    model = ViTSegmentation(image_size=224, num_classes=18)
    num_params = sum([p.numel() for p in model.parameters()])
    print(f"params: {num_params/1e6:.2f} M")


if __name__ == "__main__":
    main()