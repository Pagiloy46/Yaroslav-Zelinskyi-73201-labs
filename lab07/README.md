# Lab 7 - Model Compression and Knowledge Distillation

## Overview

This laboratory investigates neural network compression techniques using the Fashion-MNIST dataset.

The main topics are magnitude pruning, structured pruning, dense and sparse model benchmarking, and knowledge distillation.

## Project Structure

```text
lab07/
├── configs/
│   └── config.yaml
├── src/
│   ├── models.py
│   ├── train_baseline.py
│   ├── pruning.py
│   ├── structured_pruning.py
│   ├── benchmark.py
│   └── distillation.py
├── results/
├── README.md
└── report.md
```

## Requirements

Install the required packages:

```bash
pip install torch torchvision pyyaml pandas matplotlib
```

## Running the Experiments

Run commands from the repository root:

```bash
python lab07/src/models.py
python lab07/src/train_baseline.py
python lab07/src/pruning.py
python lab07/src/structured_pruning.py
python lab07/src/benchmark.py
python lab07/src/distillation.py
```

## Models

### Teacher

A large CNN with approximately 1.26 million parameters.

### Student

A compact CNN with 80,602 parameters, trained using knowledge distillation.

## Pruning

The project includes:

- Global magnitude pruning.
- Layer-wise magnitude pruning.
- Structured filter pruning using L2 norms.
- Fine-tuning after pruning.

## Knowledge Distillation

The Student model learns from the Teacher model using softened predictions.

Configuration:

- Temperature: 4
- Alpha: 0.5
- Training epochs: 3

## Evaluation Metrics

- Classification accuracy.
- Number of parameters.
- Model file size.
- Inference latency.
- Weight sparsity.

## Results

See `report.md` for the methodology, comparison tables, and conclusions.

Numerical estimates are identified separately from measured results.