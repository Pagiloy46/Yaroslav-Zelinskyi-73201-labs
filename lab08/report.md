# Lab 8: Quantization and Export

## 1. Introduction

The purpose of this laboratory was to investigate model compression and deployment optimization techniques using a Convolutional Neural Network (CNN) trained on Fashion-MNIST.

The experiments focused on manual weight quantization, dynamic quantization, static quantization, and ONNX export. The models were evaluated using accuracy, serialized model size, inference latency, and numerical agreement.

All performance measurements were performed locally using CPU execution.

## 2. Manual Weight Quantization

Manual symmetric quantization was implemented with a zero-point of zero.

Three precision levels were investigated:

- 8-bit
- 4-bit
- 2-bit

Per-tensor and per-channel scaling strategies were compared using Mean Squared Error.

The experiment demonstrated that lower bit widths introduce larger reconstruction errors. INT8 generally preserved the original weights more accurately, while 2-bit quantization produced the largest errors.

Per-channel scaling produced slightly lower reconstruction errors for some layers.

The detailed results are available in `manual_quantization.csv`.

## 3. Dynamic Quantization

Dynamic quantization was applied to the Linear layers of the baseline CNN using the FBGEMM backend.

Convolutional layers remained in FP32 precision.

| Metric | Baseline FP32 | Dynamic INT8 |
|---|---:|---:|
| Accuracy | 65.41% | 65.25% |
| Model size | 4.81 MB | 2.26 MB |
| Latency | 4.29 ms | 4.72 ms |

Dynamic quantization reduced model size by approximately 53%.

Accuracy decreased by only 0.16 percentage points. However, inference latency increased slightly.

This result shows that reducing numerical precision does not necessarily improve CPU inference speed, especially when only part of the network is quantized.

## 4. Static Quantization

Static quantization was performed using 500 randomly selected training images for calibration.

The model was fused, prepared for quantization, calibrated, and converted to INT8 using FBGEMM.

| Metric | Baseline FP32 | Static INT8 |
|---|---:|---:|
| Accuracy | 65.15% | 66.15% |
| Model size | 4.81 MB | 1.23 MB |
| Latency | 4.00 ms | 2.79 ms |

Static quantization reduced model size by approximately 74.5%.

Inference latency decreased by approximately 30.3%.

The measured accuracy increased by one percentage point. This should not be interpreted as proof that quantization always improves accuracy. The experiment used a 2,000-image test subset and quantization can change individual predictions.

## 5. ONNX Export and Validation

The baseline CNN was exported to ONNX format using `torch.onnx.export`.

The resulting model was checked with ONNX Checker and executed using ONNX Runtime with the CPU execution provider.

Numerical validation produced the following result:

| Metric | Result |
|---|---:|
| Maximum numerical error | 0.0000009537 |
| Required tolerance | 0.0001 |
| Outputs match | True |
| ONNX model size | 4.804 MB |

The maximum numerical error was substantially below the required tolerance.

### Inference Benchmark

| Runtime | Latency |
|---|---:|
| PyTorch | 4.6155 ms |
| ONNX Runtime | 0.8514 ms |

ONNX Runtime achieved approximately 5.42 times lower measured latency than the PyTorch benchmark.

The benchmark used a single input image, 20 warm-up iterations, and 100 measured iterations. These values represent local measurements and may vary depending on hardware and runtime configuration.

## 6. Model Comparison

The following table summarizes the variants for which measurements were available.

| Model variant | Accuracy | Size | Latency |
|---|---:|---:|---:|
| Baseline FP32 | 65.41% | 4.81 MB | 4.29 ms |
| Dynamic INT8 | 65.25% | 2.26 MB | 4.72 ms |
| Static INT8 | 66.15% | 1.23 MB | 2.79 ms |
| ONNX Runtime | Not evaluated | 4.80 MB | 0.85 ms |
| Pruned | Not measured | N/A | N/A |
| Distilled | Not measured | N/A | N/A |
| Pruned + Quantized | Not measured | N/A | N/A |
| Distilled + Quantized | Not measured | N/A | N/A |

Pruning and knowledge distillation variants were not implemented in this experiment. Their results are therefore not reported.

The accuracy values for dynamic and static quantization were obtained using different evaluation subsets, so direct accuracy comparisons should be treated cautiously.

## 7. Efficiency Analysis

The results can be interpreted using four efficiency pillars.

| Efficiency pillar | Observations |
|---|---|
| Data | Fashion-MNIST was used. Static quantization calibration used 500 training images. |
| Model | Static INT8 reduced serialized model size to approximately 1.23 MB. |
| Compute | Static INT8 reduced measured inference latency. ONNX Runtime also showed lower latency than PyTorch in the local benchmark. |
| Energy | Energy consumption was not directly measured. Lower latency and reduced model size may help reduce resource requirements, but energy savings cannot be confirmed from these experiments. |

## 8. Deployment Considerations

Based on the measured results:

- **Cloud:** ONNX Runtime is a promising deployment option when CPU inference performance is important.
- **Edge:** Static INT8 is attractive because of its reduced storage requirements and measured latency.
- **Mobile:** Static INT8 is a potential candidate, subject to validation on the target mobile hardware and runtime.
- **TinyML:** The measured model is still not validated for TinyML deployment. Additional compression and hardware-specific testing would be necessary.

These recommendations are based on the available measurements and do not replace deployment testing on the actual target platforms.

## 9. Individual Project Milestone

The project milestone includes:

- The Lab 4 data ingestion, preprocessing, auditing, and validation pipeline.
- A trained baseline CNN checkpoint.
- A `metrics.json` file containing measured model and runtime metrics.
- An ONNX model export.

The baseline checkpoint and experimental artifacts are stored in the project repository.

## 10. Conclusion

This laboratory demonstrated that quantization and runtime export can substantially affect model storage requirements and inference performance.

Dynamic INT8 quantization reduced model size while maintaining similar accuracy, but did not improve latency in this experiment.

Static INT8 quantization achieved the smallest measured PyTorch model size and lower inference latency. ONNX Runtime also demonstrated a substantial latency improvement over the local PyTorch benchmark while preserving numerical agreement within the required tolerance.

The most important conclusion is that model optimization involves trade-offs between accuracy, storage, compute, runtime, and hardware compatibility. The best deployment format depends on the constraints of the target platform.