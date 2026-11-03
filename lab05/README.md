# Lab 5: Frameworks and Automatic Differentiation

## Overview

This laboratory focuses on automatic differentiation, computational graphs, backpropagation, and PyTorch performance.

A custom scalar automatic differentiation engine was implemented and tested against PyTorch Autograd and numerical gradient estimates.

## Project Structure

```text
lab05/
├── results/
├── src/
│   ├── engine.py
│   ├── gradient_check.py
│   ├── train_model.py
│   └── pytorch_analysis.py
├── README.md
└── report.md
```

## Requirements

- Python 3
- NumPy
- PyTorch
- scikit-learn

Install dependencies:

```bash
pip install numpy torch scikit-learn
```

## How to Run

Run the scripts from the project root:

```bash
python lab05/src/gradient_check.py
python lab05/src/train_model.py
python lab05/src/pytorch_analysis.py
```

`engine.py` contains the custom automatic differentiation implementation and is imported by the other scripts.

## Implemented Features

### 1. Automatic Differentiation Engine
- Value nodes with data and gradient storage.
- Computational graph construction.
- Addition, multiplication, power, and ReLU operations.
- Backward pass using topological sorting.
- Gradient accumulation.

### 2. Gradient Validation
- Comparison with PyTorch Autograd.
- Numerical gradient estimation using finite differences.
- Maximum absolute error below the required tolerance of `1e-6`.

### 3. Neural Network Training
- `make_moons` dataset.
- Neural network architecture: 2 → 8 → 1.
- ReLU and Sigmoid activations.
- Binary Cross-Entropy loss.
- Performance comparison over five runs.

### 4. PyTorch Analysis
- Computation graph inspection using `torch.jit.trace`.
- Eager versus JIT Trace execution.
- `torch.compile` availability check.
- Contiguous and non-contiguous tensor comparison.

## Benchmark Results

### Custom Engine vs PyTorch

| Metric | Custom Engine | PyTorch |
|---|---:|---:|
| Median training time | 4.82358 s | 0.02767 s |
| Final loss | 0.65017 | 0.56558 |

PyTorch was approximately 174.31 times faster in this experiment.

### Execution Modes

| Mode | Median time |
|---|---:|
| Eager | 0.00002885 s |
| JIT Trace | 0.00002220 s |
| `torch.compile` | Unavailable |

`torch.compile` could not run because a working C++ compiler was not available in the environment.

### Tensor Layout

| Property | Contiguous | Non-contiguous |
|---|---|---|
| Shape | 1024 × 1024 | 1024 × 1024 |
| Stride | (1024, 1) | (1, 1024) |
| Median time | 0.00240515 s | 0.00170695 s |

The non-contiguous tensor was faster in the tested operation. This result is specific to the experiment and hardware.

## Report

See [report.md](report.md) for the detailed methodology, implementation description, benchmark results, and conclusions.