
import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    """Large CNN model for Fashion-MNIST."""

    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 384),
            nn.ReLU(),
            nn.Linear(384, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class StudentCNN(nn.Module):
    """Small CNN model for knowledge distillation."""

    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 48),
            nn.ReLU(),
            nn.Linear(48, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


if __name__ == "__main__":
    baseline = BaselineCNN()
    student = StudentCNN()

    baseline_params = sum(p.numel() for p in baseline.parameters())
    student_params = sum(p.numel() for p in student.parameters())

    print(f"Baseline parameters: {baseline_params:,}")
    print(f"Student parameters: {student_params:,}")

    x = torch.randn(1, 1, 28, 28)

    print("Baseline output:", baseline(x).shape)
    print("Student output:", student(x).shape)
