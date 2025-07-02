import torch
import torchvision.transforms as T
import PIL.Image
from typing import List


size = (224, 224)


class ResizeWithPadding:
    def __init__(self, target_size: int = 224, fill: int = 0, mode: str = "RGB") -> None:
        self.target_size = target_size
        self.fill = fill
        self.mode = mode

    def __call__(self, image: PIL.Image) -> PIL.Image:
        original_width, original_height = image.size
        aspect_ratio = original_width / original_height
        if aspect_ratio > 1:
            new_width = self.target_size
            new_height = int(self.target_size / aspect_ratio)
        else:
            new_height = self.target_size
            new_width = int(self.target_size * aspect_ratio)
        resized_image = image.resize((new_width, new_height), PIL.Image.BICUBIC if self.mode == "RGB" else PIL.Image.NEAREST)
        delta_w = self.target_size - new_width
        delta_h = self.target_size - new_height
        padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
        padded_image = PIL.Image.new(self.mode, (self.target_size, self.target_size), self.fill)
        padded_image.paste(resized_image, (padding[0], padding[1]))
        return padded_image


def get_transform(mean: List[float], std: List[float]) -> T.Compose:
    return T.Compose([
        ResizeWithPadding(),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

mask_transform = T.Compose([
    ResizeWithPadding(mode="L"),
    T.ToTensor(),
    T.Lambda(lambda x: (x * 255).long()),
])


class EMA:
    def __init__(self, alpha: float = 0.9) -> None:
        self.value = None
        self.alpha = alpha
    
    def __call__(self, value: float) -> float:
        if self.value is None:
            self.value = value
        else:
            self.value = self.alpha * self.value + (1 - self.alpha) * value
        return self.value