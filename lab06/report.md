# Lab 6: Training Pipelines and Project Proposal

## 1. Objective

The goal of this laboratory work was to build a configurable PyTorch training pipeline, analyze training performance, and investigate different training configurations.

The experiments used the Fashion-MNIST dataset and a Convolutional Neural Network (CNN).

## 2. Training Pipeline

The training process was configured using a YAML file.

The configuration included:
- Random seed: 42
- Batch size: 128
- Number of workers: 2
- Optimizer: Adam
- Learning rate: 0.001
- Scheduler: StepLR
- Number of epochs: 10

The model was trained using 55,000 training images and 5,000 validation images. The official test set contained 10,000 images.

### Training Results

| Metric | Result |
|---|---:|
| Training accuracy | 92.33% |
| Validation accuracy | 91.04% |
| Test accuracy | 90.43% |
| Test loss | 0.2655 |

The model achieved 90.43% test accuracy. This was slightly better than the CNN result from Lab 3 (89.68%).

## 3. Performance Profiling

The training process was analyzed using `torch.profiler`.

| Operation | Time (s) | Percentage |
|---|---:|---:|
| Data loading | 1.2996 | 6.65% |
| Forward pass | 6.9389 | 35.53% |
| Backward pass | 9.3768 | 48.01% |
| Optimizer step | 1.9171 | 9.81% |
| Total | 19.5324 | 100% |

The backward pass took the most time. This means that gradient calculation was the main computational bottleneck in this experiment.

## 4. DataLoader Workers Experiment

Different values of `num_workers` were tested.

| Workers | Time (s) | Throughput (samples/s) |
|---:|---:|---:|
| 0 | 7.020 | 7835.10 |
| 1 | 12.917 | 4258.03 |
| 2 | 13.761 | 3996.74 |
| 4 | 16.461 | 3341.17 |
| 8 | 26.195 | 2099.65 |

The best result was achieved with `num_workers=0`.

Increasing the number of workers made data loading slower on the tested system. This shows that more workers do not always improve performance.

## 5. Batch Size Experiment

Batch sizes from 32 to 512 were tested.

| Batch size | Training time (s) | Throughput (samples/s) | Peak process RAM (MB) | Test accuracy |
|---:|---:|---:|---:|---:|
| 32 | 359.51 | 764.92 | 420.48 | 90.42% |
| 64 | 290.11 | 947.91 | 426.25 | 90.08% |
| 128 | 254.72 | 1079.60 | 442.24 | 89.18% |
| 256 | 224.98 | 1223.67 | 471.92 | 88.82% |
| 512 | 143.50 | 1917.48 | 511.68 | 87.80% |

Larger batch sizes increased throughput and reduced training time. However, memory usage also increased, while test accuracy decreased slightly.

## 6. Optimizers and Learning Rate Schedulers

A separate experiment was prepared to compare SGD with momentum and Adam using StepLR, CosineAnnealingLR, and ReduceLROnPlateau.

**Results:** Add the measured results after running `scheduler_study.py`.

## 7. Gradient Accumulation

Gradient accumulation was implemented to compare a batch size of 256 with four micro-batches of 64.

Both approaches use an effective batch size of 256. The experiment uses the same model and training dataset subset.

**Results:** Add the measured results after running `gradient_accumulation.py`.

## 8. Checkpoint and Resume Training

A checkpoint-based training procedure was implemented.

The training process saves the model weights, optimizer state, current epoch, and training history. It is designed to interrupt training after epoch 3 and continue from the saved checkpoint.

**Results:** Add the execution results after running `resume_training.py`.

## 9. Project Proposal

The proposed individual project is a lightweight Fashion-MNIST image classifier for devices with limited computing resources.

The target requirements are:
- Inference latency: no more than 50 ms.
- Model size: no more than 5 MB.
- Classification of 10 clothing categories.

Model quantization and architecture optimization will be investigated to reduce the model size and inference time.

## 10. Conclusion

In this laboratory work, a YAML-configurable training pipeline was implemented and tested. The CNN achieved 90.43% test accuracy.

Performance profiling showed that the backward pass was the most time-consuming operation. The DataLoader experiment showed that zero workers performed best on the tested system. The batch-size experiment demonstrated a trade-off between throughput, memory usage, and accuracy.

The laboratory also included implementations for comparing optimizers and schedulers, gradient accumulation, and checkpoint-based training resumption.