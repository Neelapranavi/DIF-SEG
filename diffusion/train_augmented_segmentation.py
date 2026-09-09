import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset.segmentation_dataset import SegmentationDataset
from diffusion.augmentation import add_gaussian_noise
from evaluation.metrics import dice_score, iou_score
from segmentation.unet import UNet


def dice_loss(logits, target, eps=1e-7):
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(1, 2, 3))
    denominator = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1 - (2 * intersection + eps) / (denominator + eps)).mean()


def train(root="dataset", epochs=20, batch_size=4, learning_rate=1e-3, noise_strength=0.10):
    base = SegmentationDataset(root=root, augment=False)
    n_val = max(1, int(len(base) * 0.2))
    n_train = len(base) - n_val
    if n_train < 1:
        raise ValueError("Dataset must contain at least two paired samples.")

    train_base, val_base = random_split(
        base, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    train_dataset = SegmentationDataset(root=root, augment=True)
    val_dataset = SegmentationDataset(root=root, augment=False)
    train_dataset.samples = [base.samples[i] for i in train_base.indices]
    val_dataset.samples = [base.samples[i] for i in val_base.indices]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)
    bce = nn.BCEWithLogitsLoss()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    best = float("inf")
    Path("models").mkdir(exist_ok=True)
    Path("evaluation_results").mkdir(exist_ok=True)
    history_path = Path("evaluation_results/diffusion_segmentation_training_history.csv")

    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_dice", "val_iou", "learning_rate"])

        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            for images, masks in train_loader:
                images, masks = images.to(device), masks.to(device)
                images = add_gaussian_noise(images, strength=noise_strength)
                optimizer.zero_grad()
                logits = model(images)
                loss = bce(logits, masks) + dice_loss(logits, masks)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * images.size(0)

            model.eval()
            val_loss = 0.0
            dice_values, iou_values = [], []
            with torch.inference_mode():
                for images, masks in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    logits = model(images)
                    val_loss += (bce(logits, masks) + dice_loss(logits, masks)).item() * images.size(0)
                    probabilities = torch.sigmoid(logits).cpu().numpy()
                    targets = masks.cpu().numpy()
                    for prediction, target in zip(probabilities[:, 0], targets[:, 0]):
                        dice_values.append(dice_score(prediction, target))
                        iou_values.append(iou_score(prediction, target))

            train_loss /= len(train_dataset)
            val_loss /= len(val_dataset)
            val_dice = sum(dice_values) / len(dice_values)
            val_iou = sum(iou_values) / len(iou_values)
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch:03d}/{epochs} | train={train_loss:.4f} | "
                f"val={val_loss:.4f} | dice={val_dice:.4f} | iou={val_iou:.4f}"
            )
            writer.writerow([epoch, train_loss, val_loss, val_dice, val_iou, current_lr])
            handle.flush()

            if val_loss < best:
                best = val_loss
                torch.save(model.state_dict(), "models/unet_diffusion_best.pt")
                print("Saved models/unet_diffusion_best.pt")

    print(f"Saved diffusion training history to {history_path}")


if __name__ == "__main__":
    train()
