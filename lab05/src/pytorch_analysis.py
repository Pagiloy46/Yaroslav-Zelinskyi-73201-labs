
import time
import statistics
import torch
import torch.nn as nn


# 1. Create a small model
class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(10, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)


def benchmark_function(func, x, runs=10, warmup=3):
    # Warm-up
    for _ in range(warmup):
        func(x)

    times = []

    for _ in range(runs):
        start = time.perf_counter()
        func(x)
        times.append(time.perf_counter() - start)

    return statistics.median(times)


def analyze_execution():
    print("===== PYTORCH COMPUTATION GRAPH =====")

    model = SmallModel()
    x = torch.randn(32, 10)

    # Eager execution
    eager_output = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", eager_output.shape)

    # Trace the model
    traced_model = torch.jit.trace(model, x)

    print("\nTraced computation graph:")
    print(traced_model.graph)

    # Compare execution modes
    print("\n===== EXECUTION MODES =====")

    eager_time = benchmark_function(model, x)
    traced_time = benchmark_function(traced_model, x)

    print(f"Eager median: {eager_time:.8f} seconds")
    print(f"JIT Trace median: {traced_time:.8f} seconds")

    # Try torch.compile
    try:
        compiled_model = torch.compile(model)

        compiled_time = benchmark_function(
            compiled_model,
            x,
            runs=10,
            warmup=3
        )

        print(f"Compiled median: {compiled_time:.8f} seconds")

    except Exception as error:
        compiled_time = None
        print("torch.compile is not available:", error)


def analyze_tensors():
    print("\n===== TENSOR CONTIGUITY =====")

    x = torch.randn(1024, 1024)

    # Create a non-contiguous tensor
    non_contiguous = x.t()

    print("Contiguous tensor:")
    print("Shape:", x.shape)
    print("Dtype:", x.dtype)
    print("Stride:", x.stride())
    print("Is contiguous:", x.is_contiguous())

    print("\nNon-contiguous tensor:")
    print("Shape:", non_contiguous.shape)
    print("Dtype:", non_contiguous.dtype)
    print("Stride:", non_contiguous.stride())
    print("Is contiguous:", non_contiguous.is_contiguous())

    # Define the same operation
    def operation(tensor):
        return (tensor * 1.5 + 2.0).relu()

    # Benchmark both tensors
    contiguous_time = benchmark_function(
        operation,
        x,
        runs=20,
        warmup=5
    )

    non_contiguous_time = benchmark_function(
        operation,
        non_contiguous,
        runs=20,
        warmup=5
    )

    print("\n===== PERFORMANCE COMPARISON =====")

    print(
        f"Contiguous median: {contiguous_time:.8f} seconds"
    )

    print(
        f"Non-contiguous median: {non_contiguous_time:.8f} seconds"
    )

    print(
        "Non-contiguous / Contiguous ratio:",
        round(non_contiguous_time / contiguous_time, 3)
    )


if __name__ == "__main__":
    torch.set_num_threads(1)

    analyze_execution()
    analyze_tensors()

    print("\nAnalysis completed!")
