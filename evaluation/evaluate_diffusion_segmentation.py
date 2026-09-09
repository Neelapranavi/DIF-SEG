import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from dataset.segmentation_dataset import SegmentationDataset
from evaluation.metrics import dice_score, iou_score
from segmentation.unet import UNet


def evaluate(root="dataset", weights="models/unet_diffusion_best.pt", batch_size=4):
    dataset = SegmentationDataset(root)
    indices = list(range(len(dataset)))
    _, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
    loader = DataLoader(Subset(dataset, test_indices), batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    dice_values, iou_values = [], []
    with torch.inference_mode():
        for images, masks in loader:
            probabilities = torch.sigmoid(model(images.to(device))).cpu().numpy()
            targets = masks.numpy()
            for prediction, target in zip(probabilities[:, 0], targets[:, 0]):
                dice_values.append(dice_score(prediction, target))
                iou_values.append(iou_score(prediction, target))

    metrics = {
        "num_test_images": len(test_indices),
        "mean_dice": float(np.mean(dice_values)),
        "mean_iou": float(np.mean(iou_values)),
        "std_dice": float(np.std(dice_values)),
        "std_iou": float(np.std(iou_values)),
        "augmentation": "gaussian_noise_diffusion_inspired",
    }
    Path("evaluation_results").mkdir(exist_ok=True)
    output = Path("evaluation_results/segmentation_diffusion_metrics.json")
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {output}")
    return metrics


if __name__ == "__main__":
    evaluate()
