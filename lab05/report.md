# Lab 5: Frameworks and Automatic Differentiation

## 1. Introduction

The objective of this laboratory was to implement a scalar automatic differentiation engine, validate its gradients against PyTorch and numerical estimates, train a small neural network, and investigate PyTorch execution modes and tensor memory layouts.

The experiments were implemented in Python using a custom automatic differentiation engine and PyTorch.

## 2. Custom Automatic Differentiation Engine

A custom `Value` class was implemented in `engine.py`.

Each node stores:
- `data` – scalar numerical value.
- `grad` – accumulated gradient.
- `_prev` – parent nodes in the computation graph.
- `_op` – operation that created the node.
- `_backward` – local gradient propagation function.

The following operations were implemented:
- Addition (`+`)
- Multiplication (`*`)
- Power (`**`)
- ReLU activation

The computation graph is constructed automatically during the forward pass.

## 3. Backward Pass

The backward pass uses topological sorting to determine the order of operations.

The algorithm:
1. Builds a topological ordering of the computation graph.
2. Initializes the output gradient to 1.
3. Traverses the nodes in reverse topological order.
4. Accumulates gradients using the chain rule.

Gradient accumulation uses addition rather than assignment, allowing multiple computational paths to contribute to the same variable's gradient.

A test function was evaluated:

\[
f=(ab+c)^2
\]

For \(a=2\), \(b=3\), and \(c=1\), the output was 49.

The calculated gradients were:

\[
\frac{\partial f}{\partial a}=42
\]

\[
\frac{\partial f}{\partial b}=28
\]

\[
\frac{\partial f}{\partial c}=14
\]

All results matched the analytical derivatives.

## 4. Gradient Validation

The custom engine was validated against PyTorch Autograd and numerical finite differences.

The tested function was:

\[
f=((ab+c)^2).ReLU()
\]

The input values were \(a=2\), \(b=3\), and \(c=1\).

### Gradient comparison

| Variable | Custom engine | PyTorch | Numerical |
|---|---:|---:|---:|
| a | 42.00000000 | 42.00000000 | 42.00000000 |
| b | 28.00000000 | 28.00000000 | 28.00000000 |
| c | 14.00000000 | 14.00000000 | 14.00000000 |

Maximum absolute differences:

- Custom engine vs PyTorch: 0.0
- Custom engine vs numerical gradients: 3.9137830754043534e-09

Both differences are below the required tolerance of \(10^{-6}\).

**Result:** All gradient checks passed.

## 5. Neural Network Training on make_moons

A small neural network was trained using the `make_moons` dataset.

### Dataset and architecture

- Dataset: `make_moons`
- Number of samples: 200
- Noise: 0.2
- Random state: 42
- Architecture: 2 → 8 → 1
- Hidden activation: ReLU
- Output activation: Sigmoid
- Loss function: Binary Cross-Entropy
- Epochs: 40
- Learning rate: 0.05
- Benchmark runs: 5

The custom engine and PyTorch implementation were benchmarked separately.

### Benchmark results

| Metric | Custom engine | PyTorch |
|---|---:|---:|
| Median training time | 4.82358 s | 0.02767 s |
| Final loss | 0.65017 | 0.56558 |

The measured speed ratio was:

\[
\frac{4.82358}{0.02767}\approx174.31
\]

The custom implementation was approximately 174 times slower than PyTorch.

This difference is expected because the custom engine performs scalar operations through Python objects, while PyTorch uses optimized tensor operations and lower-level implementations.

PyTorch also achieved a lower final loss in this experiment.

## 6. PyTorch Computation Graph and Execution Modes

A small PyTorch model was created with the architecture:

\[
10 \rightarrow 16 \rightarrow 1
\]

The input tensor had shape `[32, 10]`, and the output tensor had shape `[32, 1]`.

The computation graph was inspected using `torch.jit.trace`.

### Execution time comparison

| Execution mode | Median time |
|---|---:|
| Eager execution | 0.00002885 s |
| JIT Trace | 0.00002220 s |
| torch.compile | Not available |

JIT Trace was approximately 23% faster than eager execution in this experiment.

The `torch.compile` experiment could not be completed because the environment did not have a working C++ compiler. PyTorch reported an `InvalidCxxCompiler` error because `g++` was unavailable.

JIT tracing was therefore used as the alternative execution mode.

## 7. Contiguous and Non-contiguous Tensors

Two tensors with the same shape and data type were compared.

The non-contiguous tensor was created using the transpose operation.

### Tensor properties

| Property | Contiguous | Non-contiguous |
|---|---|---|
| Shape | [1024, 1024] | [1024, 1024] |
| Data type | float32 | float32 |
| Stride | (1024, 1) | (1, 1024) |
| Contiguous | True | False |

The same operation was benchmarked on both tensors:

```python
(tensor * 1.5 + 2.0).relu()
```

### Performance results

| Tensor type | Median time |
|---|---:|
| Contiguous | 0.00240515 s |
| Non-contiguous | 0.00170695 s |

The measured ratio was:

\[
\frac{T_{non-contiguous}}{T_{contiguous}}=0.71
\]

In this experiment, the non-contiguous tensor was faster.

This result is specific to the tested operation, tensor size, and hardware. Non-contiguous tensors are not necessarily faster in general because performance depends on memory access patterns and the operations being performed.

## 8. Conclusion

The laboratory demonstrated the fundamental principles of automatic differentiation and PyTorch execution.

The custom engine successfully constructed computation graphs and propagated gradients using reverse-mode automatic differentiation. Its gradients matched PyTorch Autograd exactly and numerical estimates within the required tolerance.

The neural network experiment demonstrated the performance difference between a scalar Python implementation and an optimized tensor framework. PyTorch was approximately 174 times faster in the measured training benchmark.

The execution-mode experiment showed that JIT Trace was slightly faster than eager execution for the tested model. `torch.compile` was unavailable because the environment lacked a working C++ compiler.

Finally, the tensor experiment demonstrated that memory layout and strides affect execution performance, although the observed result depends on the specific operation and hardware.

Overall, the laboratory provided practical experience with computation graphs, automatic differentiation, gradient validation, neural network training, execution modes, and tensor memory layouts.
