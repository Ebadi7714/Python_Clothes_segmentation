from typing import List, Tuple, Callable
from pathlib import Path
import datasets
import torch
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    def __init__(
        self,
        dataset: datasets.Dataset,
        train: bool = True,
        transform: Callable = None,
        target_transform: Callable = None,
        test_size: float = 0.25,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.test_size = test_size

        total_size = len(dataset)
        indices = list(range(total_size))
        split = int(self.test_size * total_size)

        if train:
            self.indices = indices[split:]
        else:
            self.indices = indices[:split]

    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.dataset[self.indices[idx]]
        image = item["image"]
        mask = item["mask"]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        return image, mask


def collate_fn(items: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    images = torch.stack([item[0] for item in items])
    masks = torch.stack([item[1] for item in items])
    return images, masks
