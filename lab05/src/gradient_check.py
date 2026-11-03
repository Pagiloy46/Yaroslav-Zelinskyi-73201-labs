
import torch

from engine import Value


def numerical_gradient(func, x, eps=1e-6):
    return (func(x + eps) - func(x - eps)) / (2 * eps)


def run_gradient_check():
    # Input values
    a = Value(2.0)
    b = Value(3.0)
    c = Value(1.0)

    # Build computation graph
    output = ((a * b + c) ** 2).relu()

    # Backward pass
    output.backward()

    # PyTorch reference
    ta = torch.tensor(2.0, requires_grad=True)
    tb = torch.tensor(3.0, requires_grad=True)
    tc = torch.tensor(1.0, requires_grad=True)

    torch_output = ((ta * tb + tc) ** 2).relu()
    torch_output.backward()

    # Numerical gradients
    def func_a(x):
        return ((x * 3.0 + 1.0) ** 2)

    def func_b(x):
        return ((2.0 * x + 1.0) ** 2)

    def func_c(x):
        return ((2.0 * 3.0 + x) ** 2)

    numerical_a = numerical_gradient(func_a, 2.0)
    numerical_b = numerical_gradient(func_b, 3.0)
    numerical_c = numerical_gradient(func_c, 1.0)

    # Collect gradients
    custom_grads = [a.grad, b.grad, c.grad]
    torch_grads = [
        ta.grad.item(),
        tb.grad.item(),
        tc.grad.item()
    ]
    numerical_grads = [
        numerical_a,
        numerical_b,
        numerical_c
    ]

    # Compare results
    print("Gradient comparison:")
    print("-" * 65)
    print(f"{'Variable':<12}{'Custom':<15}{'PyTorch':<15}{'Numerical':<15}")

    for name, custom, pytorch, numerical in zip(
        ["a", "b", "c"],
        custom_grads,
        torch_grads,
        numerical_grads
    ):
        print(
            f"{name:<12}{custom:<15.8f}"
            f"{pytorch:<15.8f}{numerical:<15.8f}"
        )

    max_difference = max(
        abs(custom - pytorch)
        for custom, pytorch in zip(custom_grads, torch_grads)
    )

    max_numerical_difference = max(
        abs(custom - numerical)
        for custom, numerical in zip(custom_grads, numerical_grads)
    )

    print("\nMaximum difference (PyTorch):", max_difference)
    print("Maximum difference (Numerical):", max_numerical_difference)

    assert max_difference < 1e-6
    assert max_numerical_difference < 1e-6

    print("\nAll gradient checks passed!")


if __name__ == "__main__":
    run_gradient_check()
