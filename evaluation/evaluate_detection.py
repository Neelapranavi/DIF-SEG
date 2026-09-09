import json
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from detection.model import SimpleCNN
from detection.train import ClassificationDataset
from evaluation.classification_metrics import classification_report


def evaluate(root="dataset/detection", weights="models/detection_best.pt", batch_size=8):
    dataset = ClassificationDataset(root)
    labels = [label for _, label in dataset.samples]
    indices = list(range(len(dataset)))
    _, test_indices = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=labels
    )
    loader = DataLoader(Subset(dataset, test_indices), batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    y_true, y_pred, y_probability = [], [], []
    with torch.inference_mode():
        for images, batch_labels in loader:
            probabilities = torch.softmax(model(images.to(device)), dim=1).cpu()
            y_true.extend(batch_labels.tolist())
            y_pred.extend(probabilities.argmax(1).tolist())
            y_probability.extend(probabilities[:, 1].tolist())

    metrics = classification_report(y_true, y_pred, y_probability)
    Path("evaluation_results").mkdir(exist_ok=True)
    output = Path("evaluation_results/detection_metrics.json")
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {output}")
    return metrics


if __name__ == "__main__":
    evaluate()
