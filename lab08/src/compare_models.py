
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "lab08" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# Actual measurements from previous experiments.
# Missing measurements are marked as N/A.
models = [
    {
        "model": "Baseline FP32",
        "accuracy": "65.41",
        "size_mb": "4.81",
        "latency_ms": "4.29",
        "status": "Measured"
    },
    {
        "model": "Pruned",
        "accuracy": "N/A",
        "size_mb": "N/A",
        "latency_ms": "N/A",
        "status": "Not measured"
    },
    {
        "model": "Distilled",
        "accuracy": "N/A",
        "size_mb": "N/A",
        "latency_ms": "N/A",
        "status": "Not measured"
    },
    {
        "model": "Dynamic INT8",
        "accuracy": "65.25",
        "size_mb": "2.26",
        "latency_ms": "4.72",
        "status": "Measured"
    },
    {
        "model": "Static INT8",
        "accuracy": "66.15",
        "size_mb": "1.23",
        "latency_ms": "2.79",
        "status": "Measured"
    },
    {
        "model": "Pruned + Quantized",
        "accuracy": "N/A",
        "size_mb": "N/A",
        "latency_ms": "N/A",
        "status": "Not measured"
    },
    {
        "model": "Distilled + Quantized",
        "accuracy": "N/A",
        "size_mb": "N/A",
        "latency_ms": "N/A",
        "status": "Not measured"
    }
]

csv_path = RESULTS / "model_comparison.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "model",
            "accuracy",
            "size_mb",
            "latency_ms",
            "status"
        ]
    )
    writer.writeheader()
    writer.writerows(models)

print("========== MODEL COMPARISON ==========")

for model in models:
    print(
        f"{model['model']:<25} "
        f"Accuracy: {model['accuracy']:<8} "
        f"Size: {model['size_mb']:<8} "
        f"Latency: {model['latency_ms']:<8} "
        f"{model['status']}"
    )

print("\nSaved:", csv_path)
