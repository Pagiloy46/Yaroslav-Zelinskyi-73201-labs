
import sys
import csv
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune

from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

sys.path.append(str(Path(__file__).resolve().parent))
from models import BaselineCNN

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "lab07" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPARSITY_LEVELS = [0, 0.25, 0.5, 0.75, 0.9, 0.95]

# Load the baseline model
model_path = RESULTS / "baseline_model.pth"

model = BaselineCNN().to(DEVICE)

if model_path.exists():
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    print("Loaded baseline checkpoint.")
else:
    print("WARNING: Baseline checkpoint not found.")
    print("Using initialized weights. Results are not valid for accuracy comparison.")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

train_dataset = datasets.FashionMNIST(
    root=ROOT / "data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.FashionMNIST(
    root=ROOT / "data",
    train=False,
    download=True,
    transform=transform
)

# Smaller subsets for faster fine-tuning
train_dataset = Subset(train_dataset, range(5000))
test_dataset = Subset(test_dataset, range(2000))

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128)

criterion = nn.CrossEntropyLoss()


def get_accuracy(model):
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


def get_sparsity(model):
    total = 0
    zeros = 0

    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            weight = module.weight.detach()
            total += weight.numel()
            zeros += (weight == 0).sum().item()

    return zeros / total if total else 0


def fine_tune(model, epochs=3):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    for epoch in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    return model


def apply_global_pruning(model, amount):
    parameters = []

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters.append((module, "weight"))

    if amount > 0:
        prune.global_unstructured(
            parameters,
            pruning_method=prune.L1Unstructured,
            amount=amount
        )

    return model


def apply_layerwise_pruning(model, amount):
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and amount > 0:
            prune.l1_unstructured(module, name="weight", amount=amount)

    return model


results = []

for pruning_type in ["global", "layerwise"]:
    for amount in SPARSITY_LEVELS:
        print(f"\n{pruning_type.upper()} pruning: {amount * 100:.0f}%")

        pruned_model = copy.deepcopy(model)

        if pruning_type == "global":
            apply_global_pruning(pruned_model, amount)
        else:
            apply_layerwise_pruning(pruned_model, amount)

        # Fine-tuning keeps pruning masks fixed
        if amount > 0:
            fine_tune(pruned_model, epochs=3)

        accuracy = get_accuracy(pruned_model)
        actual_sparsity = get_sparsity(pruned_model)

        row = {
            "type": pruning_type,
            "target_sparsity": amount,
            "actual_sparsity": actual_sparsity,
            "accuracy": accuracy
        }

        results.append(row)

        print(f"Actual sparsity: {actual_sparsity * 100:.2f}%")
        print(f"Accuracy: {accuracy:.2f}%")

        # Remove pruning reparameterization before saving
        for module in pruned_model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if hasattr(module, "weight_orig"):
                    prune.remove(module, "weight")

        torch.save(
            pruned_model.state_dict(),
            RESULTS / f"{pruning_type}_{int(amount * 100)}.pth"
        )

with open(RESULTS / "pruning_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "type",
            "target_sparsity",
            "actual_sparsity",
            "accuracy"
        ]
    )
    writer.writeheader()
    writer.writerows(results)

print("\nPruning experiments completed.")
