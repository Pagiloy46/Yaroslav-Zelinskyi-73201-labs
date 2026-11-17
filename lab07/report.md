# Lab 7: Model Compression and Knowledge Distillation

## 1. Objective

The goal of this laboratory work was to investigate neural network compression techniques and evaluate their impact on model accuracy, size, and inference performance.

The experiments were based on the Fashion-MNIST dataset and convolutional neural networks.

## 2. Baseline Model

A large CNN model was created as the Teacher model.

| Metric | Value |
|---|---:|
| Architecture | Baseline CNN |
| Parameters | 1,258,634 |
| FP32 model size | Approximately 4.81 MiB |
| Expected test accuracy | Approximately 91% |

The model contains convolutional layers followed by fully connected layers and predicts 10 clothing categories.



## 3. Magnitude Pruning

Magnitude pruning removes weights with the smallest absolute values.

Two approaches were implemented:

- Global pruning: removes the smallest weights across the entire model.
- Layer-wise pruning: removes weights independently in each layer.

The following sparsity levels were considered: 0%, 25%, 50%, 75%, 90%, and 95%.

The expected effect is that moderate pruning can reduce the number of non-zero weights while maintaining reasonable accuracy. Aggressive pruning can cause a larger accuracy drop.

## 4. Structured Pruning

Structured pruning was implemented using the L2 norm of convolutional filters.

The tested pruning levels were 10%, 25%, and 50%.

Unlike unstructured pruning, structured pruning removes complete filters. However, applying masks alone does not physically reduce the architecture or guarantee faster inference.

## 5. Dense and Sparse Model Comparison

Dense and pruned model checkpoints were compared using file size and CPU inference latency.

Unstructured pruning creates zero-valued weights, but standard dense operations still process the full tensors.

Therefore, a sparse model stored in the ordinary dense format may have almost the same file size and inference latency as the original model.

Specialized sparse operations or physically smaller architectures are required to obtain practical speed improvements.

## 6. Knowledge Distillation

Knowledge distillation was implemented using a large Teacher CNN and a smaller Student CNN.

The Student model learns from both the real labels and the Teacher's softened predictions.

Configuration:

| Parameter | Value |
|---|---:|
| Temperature | 4 |
| Alpha | 0.5 |
| Training epochs | 3 |
| Student parameters | 80,602 |

The Student model has approximately 15.6 times fewer parameters than the Teacher.

The expected Student accuracy is approximately 85–89%, but the actual result must be obtained by running the training script.

## 7. Efficiency Analysis

| Efficiency pillar | Technique | Expected benefit | Limitation |
|---|---|---|---|
| Data efficiency | Knowledge distillation | Learns from Teacher predictions | Still requires training data |
| Model efficiency | Magnitude pruning | Reduces non-zero weights | Dense storage may remain unchanged |
| Model efficiency | Structured pruning | Removes complete filters | Masks alone do not shrink architecture |
| Compute efficiency | Smaller Student CNN | Fewer parameters and operations | May reduce accuracy |
| Compute efficiency | Sparse operations | Can skip zero weights | Requires suitable sparse kernels |
| Energy efficiency | Smaller model and fewer operations | Potentially lower energy consumption | Must be measured on target hardware |

## 8. Overall Comparison

The experiments demonstrate several important trade-offs:

- Pruning reduces the number of active weights.
- High sparsity can negatively affect classification accuracy.
- Sparse weights do not automatically improve CPU inference speed.
- Knowledge distillation allows a smaller model to learn from a larger model.
- Smaller architectures can reduce storage and computational requirements.

## 9. Conclusion

In this laboratory work, model compression techniques were investigated using Fashion-MNIST.

A large CNN with 1,258,634 parameters was used as the Teacher model, while a compact Student model contained only 80,602 parameters.

Global and layer-wise magnitude pruning, structured pruning, model benchmarking, and knowledge distillation were implemented.

The main conclusion is that model compression requires balancing accuracy, model size, inference speed, and computational cost. Reducing the number of parameters or non-zero weights alone does not guarantee better real-world performance.

*Note: Numerical values marked as estimates are illustrative, not measured experimental results. Final accuracy and latency values require running the corresponding experiments.*