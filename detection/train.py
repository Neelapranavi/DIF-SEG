from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

from detection.model import SimpleCNN


class ClassificationDataset(Dataset):
    """Folder dataset: root/normal and root/abnormal."""

    def __init__(self, root="dataset/detection", size=224):
        self.root = Path(root)
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
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
        return self.transform(Image.open(path)), label


def train(root="dataset/detection", epochs=10, batch_size=8, learning_rate=1e-3):
    dataset = ClassificationDataset(root)
    n_val = max(1, int(len(dataset) * 0.2))
    n_train = len(dataset) - n_val
    if n_train < 1:
        raise ValueError("Detection dataset needs at least two images.")
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    best = 0.0
    Path("models").mkdir(exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = total = 0
        with torch.inference_mode():
            for images, labels in val_loader:
                predictions = model(images.to(device)).argmax(1).cpu()
                correct += (predictions == labels).sum().item()
                total += labels.numel()
        accuracy = correct / total
        print(f"Epoch {epoch:03d}/{epochs} | val_accuracy={accuracy:.4f}")
        if accuracy >= best:
            best = accuracy
            torch.save(model.state_dict(), "models/detection_best.pt")


if __name__ == "__main__":
    train()
