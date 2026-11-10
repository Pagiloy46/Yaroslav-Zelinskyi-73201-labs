# Lab 6 - Training Pipelines and Project Proposal

## Overview

This laboratory focuses on building a configurable PyTorch training pipeline and analyzing model training performance.

The experiments use the Fashion-MNIST dataset and a Convolutional Neural Network (CNN).

## Project Structure

```text
lab06/
├── configs/
│   └── config.yaml
├── src/
│   ├── train.py
│   ├── profiler.py
│   ├── workers_study.py
│   ├── batch_size_study.py
│   ├── scheduler_study.py
│   ├── gradient_accumulation.py
│   └── resume_training.py
├── results/
├── project_proposal.md
├── report.md
└── README.md
```

## Requirements

Install the required packages:

```bash
pip install torch torchvision pyyaml mlflow pandas matplotlib psutil
```

## Running the Experiments

Run commands from the repository root:

```bash
python lab06/src/train.py
python lab06/src/profiler.py
python lab06/src/workers_study.py
python lab06/src/batch_size_study.py
python lab06/src/scheduler_study.py
python lab06/src/gradient_accumulation.py
python lab06/src/resume_training.py
```

## Configuration

Training parameters are stored in:

```text
lab06/configs/config.yaml
```

The configuration controls the dataset, batch size, number of workers, model, optimizer, learning rate, scheduler, and number of epochs.

## Main Results

- Test accuracy: 90.43%.
- Best DataLoader configuration: `num_workers=0`.
- Highest measured throughput in the batch-size experiment: 1917.48 samples/s at batch size 512.
- Main profiling bottleneck: backward pass (48.01%).

Detailed experiment results are available in `report.md`.

## Tools

- Python
- PyTorch
- Torchvision
- PyYAML
- MLflow
- Matplotlib
- Pandas
- PyTorch Profiler