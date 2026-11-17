
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import time
import csv

from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent))
from models import BaselineCNN


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "lab07" / "configs" / "config.yaml"
RESULTS_DIR = ROOT / "lab07" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

torch.manual_seed(config["project"]["seed"])

device_name = config["project"]["device"]
if device_name == "auto":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(device_name)

print("Device:", device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

train_dataset = datasets.FashionMNIST(
    root=ROOT / config["data"]["root"],
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.FashionMNIST(
    root=ROOT / config["data"]["root"],
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config["data"]["batch_size"],
    shuffle=True,
    num_workers=config["data"]["num_workers"]
)

test_loader = DataLoader(
    test_dataset,
    batch_size=config["data"]["batch_size"],
    shuffle=False,
    num_workers=config["data"]["num_workers"]
)

model = BaselineCNN(config["model"]["num_classes"]).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=config["training"]["learning_rate"]
)

epochs = config["training"]["epochs"]
history = []

print("\nStarting Baseline CNN training...\n")

for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    start_time = time.perf_counter()

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_time = time.perf_counter() - start_time
    train_accuracy = 100 * correct / total
    average_loss = total_loss / len(train_loader)

    history.append({
        "epoch": epoch + 1,
        "loss": average_loss,
        "accuracy": train_accuracy,
        "time": epoch_time
    })

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Loss: {average_loss:.4f} "
        f"Accuracy: {train_accuracy:.2f}% "
        f"Time: {epoch_time:.2f}s"
    )

# Test evaluation
model.eval()
correct = 0
total = 0
test_loss = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        test_loss += loss.item()
        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

test_accuracy = 100 * correct / total
test_loss /= len(test_loader)

# Model size
model_path = RESULTS_DIR / "baseline_model.pth"
torch.save(model.state_dict(), model_path)

model_size_mb = model_path.stat().st_size / (1024 * 1024)
parameters = sum(p.numel() for p in model.parameters())

with open(RESULTS_DIR / "baseline_history.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["epoch", "loss", "accuracy", "time"]
    )
    writer.writeheader()
    writer.writerows(history)

print("\n========== BASELINE RESULTS ==========")
print(f"Test accuracy: {test_accuracy:.2f}%")
print(f"Test loss: {test_loss:.4f}")
print(f"Parameters: {parameters:,}")
print(f"Model size: {model_size_mb:.2f} MB")
print(f"Checkpoint: {model_path}")
print("======================================")
