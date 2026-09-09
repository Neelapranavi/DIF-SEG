from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    """Paired grayscale image/mask dataset.

    Expected layout:
        root/images/*.png
        root/masks/*.png
    Matching files are paired by filename stem.
    """

    def __init__(self, root="dataset", size=(256, 256), augment=False):
        self.root = Path(root)
        self.size = size
        self.augment = augment
        image_dir = self.root / "images"
        mask_dir = self.root / "masks"
        self.samples = []
        for image_path in sorted(image_dir.glob("*")):
            mask_path = mask_dir / image_path.name
            if mask_path.exists():
                self.samples.append((image_path, mask_path))

        if not self.samples:
            raise FileNotFoundError(
                f"No paired samples found. Add matching files to {image_dir} and {mask_dir}."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]
        image = Image.open(image_path).convert("L").resize(self.size, Image.Resampling.BILINEAR)
        mask = Image.open(mask_path).convert("L").resize(self.size, Image.Resampling.NEAREST)

        if self.augment and torch.rand(()) < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        image = np.asarray(image, dtype=np.float32) / 255.0
        mask = (np.asarray(mask, dtype=np.float32) / 255.0 > 0.5).astype(np.float32)
        return torch.from_numpy(image[None]), torch.from_numpy(mask[None])
