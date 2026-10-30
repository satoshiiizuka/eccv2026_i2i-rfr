from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class PairedDataset(Dataset):
    def __init__(self, input_root, gt_root, crop_size=256):
        self.input_root = Path(input_root)
        self.gt_root = Path(gt_root)
        self.paths = sorted(p.relative_to(self.input_root)
                            for p in self.input_root.rglob("*.png"))
        if not self.paths:
            raise FileNotFoundError(f"No PNG images found in {self.input_root}")
        for path in self.paths:
            if not (self.gt_root / path).is_file():
                raise FileNotFoundError(f"GT not found: {self.gt_root / path}")

        self.to_tensor = T.ToTensor()
        self.augment = T.Compose([
            T.RandomCrop(crop_size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
        ])

    def __len__(self):
        return len(self.paths) * 100

    def __getitem__(self, index):
        path = self.paths[index % len(self.paths)]
        with Image.open(self.input_root / path) as image:
            input = self.to_tensor(image.convert("RGB"))
        with Image.open(self.gt_root / path) as image:
            gt = self.to_tensor(image.convert("RGB"))
        if input.shape != gt.shape:
            raise ValueError(f"Input/GT sizes differ: {path}")

        # Treat Input and GT as one 6-channel image to synchronize augmentation.
        pair = self.augment(torch.cat((input, gt), dim=0))
        if torch.rand(()) < 0.5:  # LEDNet's random transpose (90-degree rotation)
            pair = pair.transpose(-2, -1)

        return pair[:3], pair[3:]  # RGB values in [0, 1]

