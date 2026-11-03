
import time
import statistics
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_moons

from engine import Value


# Sigmoid operation for custom autograd engine
def sigmoid_value(x):
    sigmoid = 1.0 / (1.0 + np.exp(-x.data))
    out = Value(sigmoid, (x,), "sigmoid")

    def _backward():
        x.grad += sigmoid * (1.0 - sigmoid) * out.grad

    out._backward = _backward
    return out


# Binary Cross-Entropy operation
def bce_value(probability, target):
    p = min(max(probability.data, 1e-7), 1.0 - 1e-7)

    loss = -(
        target * np.log(p)
        + (1.0 - target) * np.log(1.0 - p)
    )

    out = Value(loss, (probability,), "BCE")

    def _backward():
        probability.grad += (
            (p - target) / (p * (1.0 - p))
        ) * out.grad

    out._backward = _backward
    return out


def create_dataset():
    X, y = make_moons(
        n_samples=200,
        noise=0.2,
        random_state=42
    )

    return X.astype(np.float32), y.astype(np.float32)


def initialize_parameters(seed=42):
    rng = np.random.default_rng(seed)

    W1 = [
        [Value(rng.normal(0, 0.5)) for _ in range(2)]
        for _ in range(8)
    ]
    b1 = [Value(0.0) for _ in range(8)]

    W2 = [
        [Value(rng.normal(0, 0.5)) for _ in range(8)]
    ]
    b2 = [Value(0.0)]

    return W1, b1, W2, b2


def custom_forward(x, parameters):
    W1, b1, W2, b2 = parameters

    hidden = []

    for i in range(8):
        activation = b1[i]

        for j in range(2):
            activation = activation + W1[i][j] * x[j]

        hidden.append(activation.relu())

    output = b2[0]

    for i in range(8):
        output = output + W2[0][i] * hidden[i]

    return sigmoid_value(output)


def train_custom(X, y, epochs=40, learning_rate=0.05):
    parameters = initialize_parameters()

    W1, b1, W2, b2 = parameters
    all_parameters = (
        [p for row in W1 for p in row]
        + b1
        + [p for row in W2 for p in row]
        + b2
    )

    for epoch in range(epochs):
        for p in all_parameters:
            p.grad = 0.0

        losses = []

        for x, target in zip(X, y):
            inputs = [Value(x[0]), Value(x[1])]
            prediction = custom_forward(inputs, parameters)
            losses.append(bce_value(prediction, float(target)))

        total_loss = losses[0]

        for loss in losses[1:]:
            total_loss = total_loss + loss

        total_loss = total_loss * (1.0 / len(losses))
        total_loss.backward()

        for p in all_parameters:
            p.data -= learning_rate * p.grad

    return total_loss.data


class TorchModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


def train_pytorch(X, y, epochs=40, learning_rate=0.05):
    torch.manual_seed(42)

    model = TorchModel()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate
    )
    criterion = nn.BCELoss()

    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y).reshape(-1, 1)

    for epoch in range(epochs):
        optimizer.zero_grad()

        predictions = model(X_tensor)
        loss = criterion(predictions, y_tensor)

        loss.backward()
        optimizer.step()

    return loss.item()


def benchmark():
    X, y = create_dataset()

    custom_times = []
    torch_times = []

    custom_losses = []
    torch_losses = []

    runs = 5

    for run in range(runs):
        start = time.perf_counter()
        custom_loss = train_custom(X, y)
        custom_times.append(time.perf_counter() - start)
        custom_losses.append(custom_loss)

        start = time.perf_counter()
        torch_loss = train_pytorch(X, y)
        torch_times.append(time.perf_counter() - start)
        torch_losses.append(torch_loss)

        print(f"Run {run + 1}/{runs} completed")

    print("\n===== BENCHMARK RESULTS =====")

    print(
        "Custom engine median:",
        round(statistics.median(custom_times), 5),
        "seconds"
    )

    print(
        "PyTorch median:",
        round(statistics.median(torch_times), 5),
        "seconds"
    )

    print(
        "Custom final loss:",
        round(statistics.median(custom_losses), 5)
    )

    print(
        "PyTorch final loss:",
        round(statistics.median(torch_losses), 5)
    )

    print(
        "Speed ratio (Custom / PyTorch):",
        round(
            statistics.median(custom_times)
            / statistics.median(torch_times),
            2
        )
    )


if __name__ == "__main__":
    benchmark()
