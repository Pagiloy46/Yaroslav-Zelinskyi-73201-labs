
import sys
import csv
import torch
import torch.nn as nn

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "lab08" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "lab07" / "src"))
from models import BaselineCNN

MODEL_PATH = ROOT / "lab07" / "results" / "baseline_model.pth"

if MODEL_PATH.exists():
    model = BaselineCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    print("Loaded trained Baseline CNN.")
else:
    print("Baseline checkpoint not found. Using initialized model.")
    model = BaselineCNN()

model.eval()


def quantize_tensor(weights, bits):
    qmax = 2 ** (bits - 1) - 1

    scale = weights.abs().max() / qmax
    scale = torch.clamp(scale, min=1e-12)

    quantized = torch.round(weights / scale).clamp(-qmax, qmax)
    reconstructed = quantized * scale

    mse = torch.mean((weights - reconstructed) ** 2).item()

    return mse


def quantize_per_channel(weights, bits):
    qmax = 2 ** (bits - 1) - 1

    # Quantize each output channel independently
    flat = weights.reshape(weights.shape[0], -1)

    scales = flat.abs().amax(dim=1, keepdim=True) / qmax
    scales = torch.clamp(scales, min=1e-12)

    quantized = torch.round(flat / scales).clamp(-qmax, qmax)
    reconstructed = quantized * scales

    mse = torch.mean((flat - reconstructed) ** 2).item()

    return mse


results = []

for name, parameter in model.named_parameters():
    if "weight" not in name:
        continue

    weights = parameter.detach().float()

    for bits in [8, 4, 2]:
        mse_tensor = quantize_tensor(weights, bits)

        mse_channel = None
        if weights.ndim >= 2:
            mse_channel = quantize_per_channel(weights, bits)

        results.append({
            "layer": name,
            "bits": bits,
            "per_tensor_mse": mse_tensor,
            "per_channel_mse": mse_channel
        })

        print(f"\nLayer: {name}")
        print(f"Bits: {bits}")
        print(f"Per-tensor MSE: {mse_tensor:.8f}")

        if mse_channel is not None:
            print(f"Per-channel MSE: {mse_channel:.8f}")

csv_path = RESULTS / "manual_quantization.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "layer",
            "bits",
            "per_tensor_mse",
            "per_channel_mse"
        ]
    )
    writer.writeheader()
    writer.writerows(results)

print("\nManual quantization completed.")
print("Results saved to:", csv_path)
