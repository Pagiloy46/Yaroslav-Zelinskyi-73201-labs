
import sys
import csv
import time
import torch
import onnx
import onnxruntime as ort
import numpy as np

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "lab08" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

sys.path.append(str(ROOT / "lab07" / "src"))
from models import BaselineCNN

MODEL_PATH = ROOT / "lab07" / "results" / "baseline_model.pth"
ONNX_PATH = RESULTS / "baseline_model.onnx"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Baseline model not found: {MODEL_PATH}"
    )

# Load PyTorch model
model = BaselineCNN()
model.load_state_dict(
    torch.load(MODEL_PATH, map_location="cpu")
)
model.eval()

dummy_input = torch.randn(1, 1, 28, 28)

# Export model to ONNX
print("Exporting model to ONNX...")

torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    input_names=["input"],
    output_names=["output"],
    opset_version=18,
    do_constant_folding=True
)

print("ONNX export completed.")

# Validate ONNX model
onnx_model = onnx.load(str(ONNX_PATH))
onnx.checker.check_model(onnx_model)

print("ONNX model validation successful.")

# Create ONNX Runtime session
session = ort.InferenceSession(
    str(ONNX_PATH),
    providers=["CPUExecutionProvider"]
)

# Compare PyTorch and ONNX outputs
test_input = torch.randn(1, 1, 28, 28)

with torch.no_grad():
    pytorch_output = model(test_input).numpy()

onnx_output = session.run(
    ["output"],
    {"input": test_input.numpy()}
)[0]

max_error = float(
    np.max(np.abs(pytorch_output - onnx_output))
)

tolerance = 1e-4
agreement = max_error <= tolerance

print("\n========== NUMERICAL VALIDATION ==========")
print(f"Maximum error: {max_error:.10f}")
print(f"Tolerance: {tolerance}")
print(f"Outputs match: {agreement}")
print("==========================================")

# Benchmark PyTorch
def benchmark_pytorch(model, sample):
    with torch.no_grad():
        for _ in range(20):
            model(sample)

        start = time.perf_counter()

        for _ in range(100):
            model(sample)

    return (time.perf_counter() - start) * 10


# Benchmark ONNX Runtime
def benchmark_onnx(session, sample):
    input_array = sample.numpy()
    input_name = session.get_inputs()[0].name

    for _ in range(20):
        session.run(None, {input_name: input_array})

    start = time.perf_counter()

    for _ in range(100):
        session.run(None, {input_name: input_array})

    return (time.perf_counter() - start) * 10


pytorch_latency = benchmark_pytorch(model, test_input)
onnx_latency = benchmark_onnx(session, test_input)

onnx_size = ONNX_PATH.stat().st_size / (1024 * 1024)

results = {
    "pytorch_latency_ms": pytorch_latency,
    "onnx_latency_ms": onnx_latency,
    "onnx_size_mb": onnx_size,
    "max_error": max_error,
    "tolerance": tolerance,
    "agreement": agreement
}

# Save results
csv_path = RESULTS / "onnx_results.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(results.keys())
    )
    writer.writeheader()
    writer.writerow(results)

print("\n========== ONNX BENCHMARK ==========")
print(f"PyTorch latency: {pytorch_latency:.4f} ms")
print(f"ONNX latency: {onnx_latency:.4f} ms")
print(f"ONNX model size: {onnx_size:.3f} MB")
print(f"Maximum error: {max_error:.10f}")
print(f"Numerical agreement: {agreement}")
print(f"Results saved: {csv_path}")
print("====================================")
