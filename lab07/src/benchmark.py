
import sys
import csv
import time
import torch

from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from models import BaselineCNN

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "lab07" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# CPU benchmark: compare standard dense kernels
DEVICE = torch.device("cpu")
WARMUP = 20
ITERATIONS = 100

baseline_path = RESULTS / "baseline_model.pth"

if not baseline_path.exists():
    raise FileNotFoundError(
        "baseline_model.pth is missing. "
        "A trained baseline checkpoint is required for this comparison."
    )


def load_model(path):
    model = BaselineCNN()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def measure_latency(model):
    sample = torch.randn(1, 1, 28, 28)

    with torch.no_grad():
        for _ in range(WARMUP):
            model(sample)

        start = time.perf_counter()

        for _ in range(ITERATIONS):
            model(sample)

        elapsed = time.perf_counter() - start

    return elapsed * 1000 / ITERATIONS


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


model_files = [
    ("Dense baseline", baseline_path),
    ("Global pruning 50%", RESULTS / "global_50.pth"),
    ("Global pruning 90%", RESULTS / "global_90.pth"),
    ("Structured pruning 25%", RESULTS / "structured_25.pth"),
    ("Structured pruning 50%", RESULTS / "structured_50.pth")
]

results = []

print("\n========== DENSE VS SPARSE BENCHMARK ==========")

for name, path in model_files:
    if not path.exists():
        print(f"Skipping {name}: checkpoint not found.")
        continue

    model = load_model(path)

    latency = measure_latency(model)
    size_mb = path.stat().st_size / (1024 * 1024)
    parameters = count_parameters(model)

    row = {
        "model": name,
        "parameters": parameters,
        "size_mb": size_mb,
        "latency_ms": latency
    }

    results.append(row)

    print(f"\n{name}")
    print(f"Parameters: {parameters:,}")
    print(f"File size: {size_mb:.3f} MB")
    print(f"Latency: {latency:.3f} ms")

with open(RESULTS / "benchmark_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["model", "parameters", "size_mb", "latency_ms"]
    )
    writer.writeheader()
    writer.writerows(results)

print("\nBenchmark completed.")
print("Results saved to benchmark_results.csv")
