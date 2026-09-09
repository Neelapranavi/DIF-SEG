from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from detection.model import SimpleCNN
from evaluation.classification_metrics import classification_report


class ClassificationDataset(Dataset):
    """Folder dataset: root/normal and root/abnormal."""

    def __init__(self, root="dataset/detection", size=224):
        self.root = Path(root)
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])
        self.samples = []
        for label, name in enumerate(("normal", "abnormal")):
            folder = self.root / name
            for path in sorted(folder.glob("*")):
                if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
                    self.samples.append((path, label))
        if not self.samples:
            raise FileNotFoundError("Add images under dataset/detection/normal and dataset/detection/abnormal.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        return self.transform(Image.open(path).convert("RGB")), label


def train(root="dataset/detection", epochs=10, batch_size=8, learning_rate=1e-3):
    dataset = ClassificationDataset(root)
    labels = [label for _, label in dataset.samples]
    if len(set(labels)) < 2:
        raise ValueError("Detection training requires both normal and abnormal classes.")

    indices = list(range(len(dataset)))
    train_indices, val_indices = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=labels
    )
    train_set, val_set = Subset(dataset, train_indices), Subset(dataset, val_indices)

    counts = Counter(labels[i] for i in train_indices)
    class_weights = torch.tensor(
        [len(train_indices) / (2 * counts.get(i, 1)) for i in range(2)],
        dtype=torch.float32,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    best_f1 = -1.0
    Path("models").mkdir(exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        for images, labels_batch in train_loader:
            images, labels_batch = images.to(device), labels_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        y_true, y_pred, y_probability = [], [], []
        with torch.inference_mode():
            for images, labels_batch in val_loader:
                probabilities = torch.softmax(model(images.to(device)), dim=1).cpu()
                predictions = probabilities.argmax(1)
                y_true.extend(labels_batch.tolist())
                y_pred.extend(predictions.tolist())
                y_probability.extend(probabilities[:, 1].tolist())

        metrics = classification_report(y_true, y_pred, y_probability)
        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"val_accuracy={metrics['accuracy']:.4f} | "
            f"val_f1={metrics['f1']:.4f} | "
            f"val_recall={metrics['recall_sensitivity']:.4f}"
        )
        if metrics["f1"] >= best_f1:
            best_f1 = metrics["f1"]
            torch.save(model.state_dict(), "models/detection_best.pt")


if __name__ == "__main__":
    train()
