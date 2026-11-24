
import sys
import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "lab07" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

sys.path.append(str(ROOT / "lab07" / "src"))
from models import BaselineCNN

torch.manual_seed(42)
device = torch.device("cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

dataset = datasets.FashionMNIST(
    ROOT / "data",
    train=True,
    download=True,
    transform=transform
)

dataset = Subset(dataset, range(5000))

loader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=True,
    num_workers=0
)

model = BaselineCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

model.train()

for images, labels in loader:
    images, labels = images.to(device), labels.to(device)

    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

print("Short training completed.")

torch.save(
    model.state_dict(),
    RESULTS / "baseline_model.pth"
)

print("Saved:", RESULTS / "baseline_model.pth")
