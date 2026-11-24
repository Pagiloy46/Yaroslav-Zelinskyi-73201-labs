# Lab 8: Quantization and Export

## Overview

This laboratory explores model quantization, numerical precision, and deployment-oriented optimization using a CNN trained on the Fashion-MNIST dataset.

The experiments include manual weight quantization, dynamic quantization, static quantization, ONNX export, and inference benchmarking.

The main objective is to compare model accuracy, storage size, inference latency, and numerical agreement across different deployment formats.

## Project Structure

```text
lab08/
├── configs/
│   └── config.yaml
├── models/
│   └── project_baseline.pth
├── src/
│   ├── manual_quantization.py
│   ├── dynamic_quantization.py
│   ├── static_quantization.py
│   ├── export_onnx.py
│   ├── compare_models.py
│   └── utils.py
├── results/
│   ├── manual_quantization.csv
│   ├── dynamic_quantization.csv
│   ├── static_quantization.csv
│   ├── onnx_results.csv
│   ├── model_comparison.csv
│   ├── metrics.json
│   └── baseline_model.onnx
├── README.md
└── report.md
```

The baseline checkpoint is stored in `models/project_baseline.pth`. Experimental outputs and measurements are stored in `results/`.

## Environment

Experiments were performed using:

- Python 3.11
- PyTorch
- Fashion-MNIST
- ONNX 1.17.0
- ONNX Runtime 1.18.0
- NumPy 1.26.4
- CPU execution
- FBGEMM quantization backend

Install the main dependencies:

```bash
pip install torch torchvision numpy pandas matplotlib pyyaml
pip install onnx==1.17.0 onnxruntime==1.18.0
```

## Experiments

### 1. Manual Quantization

Weights were quantized using symmetric quantization with a zero-point of zero.

The following bit widths were evaluated:

- INT8
- 4-bit
- 2-bit

Per-tensor and per-channel scaling were compared using Mean Squared Error (MSE).

Results are stored in `results/manual_quantization.csv`.

### 2. Dynamic Quantization

Dynamic INT8 quantization was applied to the fully connected layers using the FBGEMM backend.

The convolutional layers remained in FP32 precision.

The experiment measured accuracy, serialized model size, and inference latency.

Results are stored in `results/dynamic_quantization.csv`.

### 3. Static Quantization

Static quantization was performed using calibration data consisting of 500 Fashion-MNIST training images.

The model was converted to INT8 using the FBGEMM backend.

Accuracy, model size, and inference latency were evaluated.

Results are stored in `results/static_quantization.csv`.

### 4. ONNX Export

The baseline model was exported to ONNX format using `torch.onnx.export`.

The exported model was validated using ONNX Checker and executed with ONNX Runtime.

Numerical agreement was evaluated by comparing PyTorch and ONNX outputs with a tolerance of 1e-4.

Results are stored in `results/onnx_results.csv`.

### 5. Model Comparison

Measured variants were compared using accuracy, model size, and inference latency.

Pruned and distilled variants, including their combined quantized versions, were not measured in this implementation.

The comparison is available in `results/model_comparison.csv`.

## Running the Experiments

Run scripts from the repository root:

```bash
python lab08/src/manual_quantization.py
python lab08/src/dynamic_quantization.py
python lab08/src/static_quantization.py
python lab08/src/export_onnx.py
python lab08/src/compare_models.py
```

## Main Results

| Model | Accuracy | Size (MB) | Latency (ms) |
|---|---:|---:|---:|
| Baseline FP32 | 65.41% | 4.81 | 4.29 |
| Dynamic INT8 | 65.25% | 2.26 | 4.72 |
| Static INT8 | 66.15% | 1.23 | 2.79 |
| ONNX Runtime | Not evaluated | 4.80 | 0.85 |

The accuracy measurements were obtained in separate experimental runs. The static quantization experiment used a 2,000-image test subset, while the dynamic quantization experiment evaluated the full test set. Therefore, accuracy values should not be interpreted as a perfectly controlled head-to-head comparison.

## Conclusion

Static INT8 quantization achieved the smallest measured model size and the lowest latency among the measured PyTorch variants. Dynamic quantization reduced storage requirements but did not improve inference latency in this experiment.

ONNX Runtime achieved substantially lower measured latency than the PyTorch execution in the local CPU benchmark, while maintaining numerical agreement within the required tolerance.

The results demonstrate that deployment optimization should consider accuracy, model size, runtime, and hardware backend together rather than relying on compression ratio alone.