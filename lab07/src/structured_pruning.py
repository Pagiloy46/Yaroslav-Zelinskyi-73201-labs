
import sys
import copy
import csv
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.optim as optim

from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

sys.path.append(str(Path(__file__).resolve().parent))
from models import BaselineCNN

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "lab07" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEVELS = [0.10, 0.25, 0.50]

# Load baseline
model = BaselineCNN().to(DEVICE)
checkpoint = RESULTS / "baseline_model.pth"

if not checkpoint.exists():
    raise FileNotFoundError(
        "baseline_model.pth not found. Train the baseline model first."
    )

model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

train_data = datasets.FashionMNIST(
    ROOT / "data", train=True, download=True, transform=transform
)
test_data = datasets.FashionMNIST(
    ROOT / "data", train=False, download=True, transform=transform
)

train_data = Subset(train_data, range(5000))
test_data = Subset(test_data, range(2000))

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128)


def evaluate(model):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total


def calculate_sparsity(model):
    total = 0
    zeros = 0

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            weights = module.weight.detach()
            total += weights.numel()
            zeros += (weights == 0).sum().item()

    return zeros / total if total else 0


def fine_tune(model, epochs=3):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

    return model


results = []

for amount in LEVELS:
    print(f"\nStructured pruning: {amount * 100:.0f}%")

    pruned_model = copy.deepcopy(model)

    # Remove entire output filters using L2 norm
    for module in pruned_model.modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(
                module,
                name="weight",
                amount=amount,
                n=2,
                dim=0
            )

    # Fine-tune while keeping pruning masks fixed
    fine_tune(pruned_model, epochs=3)

    accuracy = evaluate(pruned_model)
    sparsity = calculate_sparsity(pruned_model)

    print(f"Convolutional weight sparsity: {sparsity * 100:.2f}%")
    print(f"Test accuracy: {accuracy:.2f}%")

    results.append({
        "target_filter_pruning": amount,
        "conv_weight_sparsity": sparsity,
        "accuracy": accuracy
    })

    # Make pruning permanent
    for module in pruned_model.modules():
        if isinstance(module, nn.Conv2d):
            prune.remove(module, "weight")

    torch.save(
        pruned_model.state_dict(),
        RESULTS / f"structured_{int(amount * 100)}.pth"
    )

with open(RESULTS / "structured_pruning.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "target_filter_pruning",
            "conv_weight_sparsity",
            "accuracy"
        ]
    )
    writer.writeheader()
    writer.writerows(results)

print("\nStructured pruning completed.")
