# Individual Project Proposal

**Project Title:** Car Make and Model Recognition Using Deep Learning

### 1. Problem

The goal of this project is to develop a deep learning model that can recognize a car's manufacturer and specific model from an image. The system will help automatically identify vehicles using computer vision.

### 2. Dataset

The project will use the CompCars dataset. It contains images of different car manufacturers and models, with additional information about vehicle viewpoints and characteristics.

A subset of approximately 100 car models will be selected for the initial experiment to make training and evaluation feasible.

### 3. Success Metrics

The model will be evaluated using:
- Top-1 classification accuracy.
- Top-5 classification accuracy.
- Inference latency.
- Model size.

The target is to achieve at least 70% Top-1 accuracy and 90% Top-5 accuracy on the selected classes.

### 4. Deployment Constraints

The trained model must satisfy the following requirements:
- Inference latency: ≤ 50 ms per image.
- Model size: ≤ 5 MB.
- CPU inference should be supported.

### 5. Planned Optimization

MobileNetV3-Small will be used as the main architecture. INT8 quantization will be applied to reduce model size and improve inference speed. The optimized model will be compared with the original model in terms of accuracy, latency, and size.

### 6. Expected Result

The expected result is a lightweight image classification system capable of recognizing different car manufacturers and models while satisfying the deployment constraints.