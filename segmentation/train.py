from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset.segmentation_dataset import SegmentationDataset
from segmentation.unet import UNet


def dice_loss(logits, target, eps=1e-7):
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(1, 2, 3))
    denominator = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1 - (2 * intersection + eps) / (denominator + eps)).mean()


def train(root="dataset", epochs=20, batch_size=4, learning_rate=1e-3, val_split=0.2):
    base = SegmentationDataset(root=root, augment=False)
    n_val = max(1, int(len(base) * val_split))
    n_train = len(base) - n_val
    if n_train < 1:
        raise ValueError("Dataset must contain at least two paired samples.")

    train_base, val_base = random_split(base, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    train_dataset = SegmentationDataset(root=root, augment=True)
    val_dataset = SegmentationDataset(root=root, augment=False)
    train_dataset.samples = [base.samples[i] for i in train_base.indices]
    val_dataset.samples = [base.samples[i] for i in val_base.indices]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    bce = nn.BCEWithLogitsLoss()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    best = float("inf")
    Path("models").mkdir(exist_ok=True)
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = bce(logits, masks) + dice_loss(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                val_loss += (bce(logits, masks) + dice_loss(logits, masks)).item() * images.size(0)

        train_loss /= len(train_dataset)
        val_loss /= len(val_dataset)
        print(f"Epoch {epoch:03d}/{epochs} | train={train_loss:.4f} | val={val_loss:.4f}")
        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), "models/unet_best.pt")
            print("Saved models/unet_best.pt")


if __name__ == "__main__":
    train()
