# Individual Project Proposal

## Project Title
**Lightweight Fashion Image Classification for Edge Devices**

## 1. Problem Statement

The goal of this project is to develop a small and efficient neural network that can classify clothing images. The model should work on devices with limited computing resources, such as laptops and mobile devices.

## 2. Dataset

The project will use the Fashion-MNIST dataset.

It contains 70,000 grayscale images divided into 10 clothing categories. Each image has a resolution of 28 × 28 pixels.

The dataset contains 60,000 training images and 10,000 test images.

## 3. Proposed Model

A lightweight Convolutional Neural Network (CNN) will be used.

The model will contain:
- Two convolutional layers.
- ReLU activation functions.
- Max pooling layers.
- A fully connected output layer with 10 classes.

The model will be based on the CNN architecture used in previous laboratory work.

## 4. Performance Requirements

The project must satisfy the following constraints:

| Metric | Requirement |
|---|---:|
| Inference latency | ≤ 50 ms |
| Model size | ≤ 5 MB |
| Number of classes | 10 |

These requirements will help ensure that the model is suitable for resource-limited devices.

## 5. Optimization Strategy

Several optimization techniques will be investigated:

- Model quantization to reduce model size.
- Efficient preprocessing of input images.
- Batch size optimization.
- Removal of unnecessary model complexity.
- Inference benchmarking on CPU.

The final model will be evaluated using accuracy, inference latency, and storage size.

## 6. Expected Results

The expected result is a compact image classification model that satisfies the latency and storage requirements while maintaining good classification accuracy.

The project will demonstrate how machine learning models can be optimized for practical deployment on devices with limited resources.

## 7. Evaluation

The model will be evaluated using:
- Test accuracy.
- Inference latency in milliseconds.
- Model size in megabytes.

The final objective is to achieve a balance between prediction quality, speed, and model size.