
import sys
import csv
import time
import torch
import torch.nn as nn
import torch.ao.quantization as tq

from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "lab08" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

sys.path.append(str(ROOT / "lab07" / "src"))
from models import BaselineCNN

torch.backends.quantized.engine = "fbgemm"


class QuantizedWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.quant = tq.QuantStub()
        self.model = model
        self.dequant = tq.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x


MODEL_PATH = ROOT / "lab07" / "results" / "baseline_model.pth"

if not MODEL_PATH.exists():
    raise FileNotFoundError("Baseline checkpoint not found.")

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

calibration_data = Subset(train_data, range(500))
test_data = Subset(test_data, range(2000))

calibration_loader = DataLoader(
    calibration_data, batch_size=64, shuffle=False
)

test_loader = DataLoader(
    test_data, batch_size=128, shuffle=False
)

# Original FP32 model
model = BaselineCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# Prepare a separate model for quantization
quant_model = BaselineCNN()
quant_model.load_state_dict(model.state_dict())
quant_model.eval()

# Correct fusion: Conv + ReLU
quant_model.features = tq.fuse_modules(
    quant_model.features,
    [["0", "1"], ["3", "4"], ["6", "7"]],
    inplace=False
)

# Correct fusion: Linear + ReLU
quant_model.classifier = tq.fuse_modules(
    quant_model.classifier,
    [["1", "2"]],
    inplace=False
)

quant_model = QuantizedWrapper(quant_model)
quant_model.eval()

quant_model.qconfig = tq.get_default_qconfig("fbgemm")

prepared = tq.prepare(quant_model, inplace=False)

print("Calibration started...")

with torch.no_grad():
    for images, _ in calibration_loader:
        prepared(images)

print("Calibration completed.")

quantized_model = tq.convert(prepared, inplace=False)
quantized_model.eval()


def evaluate(model):
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total


def latency(model):
    sample = torch.randn(1, 1, 28, 28)

    with torch.no_grad():
        for _ in range(20):
            model(sample)

        start = time.perf_counter()

        for _ in range(100):
            model(sample)

        return (time.perf_counter() - start) * 10


fp32_path = RESULTS / "static_baseline_fp32.pth"
int8_path = RESULTS / "static_quantized_int8.pth"

torch.save(model.state_dict(), fp32_path)
torch.save(quantized_model.state_dict(), int8_path)

results = [
    {
        "model": "Baseline FP32",
        "accuracy": evaluate(model),
        "size_mb": fp32_path.stat().st_size / (1024 * 1024),
        "latency_ms": latency(model)
    },
    {
        "model": "Static INT8",
        "accuracy": evaluate(quantized_model),
        "size_mb": int8_path.stat().st_size / (1024 * 1024),
        "latency_ms": latency(quantized_model)
    }
]

with open(RESULTS / "static_quantization.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["model", "accuracy", "size_mb", "latency_ms"]
    )
    writer.writeheader()
    writer.writerows(results)

for row in results:
    print(row)

print("\nStatic quantization completed.")
