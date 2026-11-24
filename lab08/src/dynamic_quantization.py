
import sys
import csv
import time
import torch
import torch.nn as nn

from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "lab08" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

sys.path.append(str(ROOT / "lab07" / "src"))
from models import BaselineCNN

DEVICE = torch.device("cpu")
BACKEND = "fbgemm"

torch.backends.quantized.engine = BACKEND

MODEL_PATH = RESULTS.parent.parent / "lab07" / "results" / "baseline_model.pth"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Trained baseline_model.pth not found. "
        "Dynamic quantization requires a trained baseline."
    )

model = BaselineCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

quantized_model = torch.quantization.quantize_dynamic(
    model,
    {nn.Linear},
    dtype=torch.qint8
)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

test_dataset = datasets.FashionMNIST(
    root=ROOT / "data",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=0
)


def evaluate(model):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total


def measure_latency(model):
    sample = torch.randn(1, 1, 28, 28)

    with torch.no_grad():
        for _ in range(20):
            model(sample)

        start = time.perf_counter()

        for _ in range(100):
            model(sample)

        return (time.perf_counter() - start) * 10


def save_state(model, path):
    torch.save(model.state_dict(), path)
    return path.stat().st_size / (1024 * 1024)


baseline_size = save_state(
    model, RESULTS / "dynamic_baseline_fp32.pth"
)

quantized_size = save_state(
    quantized_model, RESULTS / "dynamic_quantized_int8.pth"
)

results = [
    {
        "model": "Baseline FP32",
        "backend": "none",
        "accuracy": evaluate(model),
        "size_mb": baseline_size,
        "latency_ms": measure_latency(model)
    },
    {
        "model": "Dynamic INT8",
        "backend": BACKEND,
        "accuracy": evaluate(quantized_model),
        "size_mb": quantized_size,
        "latency_ms": measure_latency(quantized_model)
    }
]

with open(RESULTS / "dynamic_quantization.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "model", "backend", "accuracy", "size_mb", "latency_ms"
        ]
    )
    writer.writeheader()
    writer.writerows(results)

for row in results:
    print("\n", row)

print("\nDynamic quantization completed.")
