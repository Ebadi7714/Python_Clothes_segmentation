from transformers import Dinov2Backbone
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.segmentation_head import SegmentationHead


class DINOSegmentationModel(nn.Module):
    def __init__(self, image_size: int = 224, num_classes: int = 18) -> None:
        super().__init__()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.image_size = image_size
        model_name = "facebook/dinov2-small"
        self.backbone = Dinov2Backbone.from_pretrained(model_name)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.segmentation_head = SegmentationHead(in_channels=384, num_classes=num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.size()
        assert height == width == self.image_size, "The image must match the size required by the DINO model"
        features = self.backbone(pixel_values=x).feature_maps[0]
        masks = self.segmentation_head(features)
        return masks


def main() -> None:
    # model = DINOSegmentationModel()
    model = SegmentationHead(384, 18)
    num_params = sum([p.numel() for p in model.parameters()])
    print(f"params: {num_params/1e6:.2f} M")


if __name__ == "__main__":
    main()