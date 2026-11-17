
import sys
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

sys.path.append(str(Path(__file__).resolve().parent))
from models import BaselineCNN, StudentCNN

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "lab07" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEMPERATURE = 4
ALPHA = 0.5
EPOCHS = 3
BATCH_SIZE = 128

torch.manual_seed(42)

# Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

train_data = datasets.FashionMNIST(
    ROOT / "data", train=True, download=True, transform=transform
)

test_data = datasets.FashionMNIST(
    ROOT / "data", train=False, download=True, transform=transform
)

# Small subsets for faster experiments
train_data = Subset(train_data, range(5000))
test_data = Subset(test_data, range(2000))

train_loader = DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
test_loader = DataLoader(
    test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

# Load trained teacher
teacher = BaselineCNN().to(DEVICE)
teacher_path = RESULTS / "baseline_model.pth"

if not teacher_path.exists():
    raise FileNotFoundError(
        "Trained baseline_model.pth is missing. "
        "Knowledge distillation requires a trained teacher model."
    )

teacher.load_state_dict(
    torch.load(teacher_path, map_location=DEVICE)
)
teacher.eval()

for parameter in teacher.parameters():
    parameter.requires_grad = False

# Create student
student = StudentCNN().to(DEVICE)

optimizer = optim.Adam(student.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


def evaluate(model):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total


history = []

print("Starting Knowledge Distillation...")
print(f"Temperature: {TEMPERATURE}")
print(f"Alpha: {ALPHA}")

for epoch in range(EPOCHS):
    student.train()
    total_loss = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        # Teacher predictions
        with torch.no_grad():
            teacher_outputs = teacher(images)

        # Student predictions
        student_outputs = student(images)

        # Soft targets
        soft_targets = F.softmax(
            teacher_outputs / TEMPERATURE, dim=1
        )

        student_log_probs = F.log_softmax(
            student_outputs / TEMPERATURE, dim=1
        )

        distillation_loss = F.kl_div(
            student_log_probs,
            soft_targets,
            reduction="batchmean"
        ) * (TEMPERATURE ** 2)

        # Normal classification loss
        student_loss = criterion(student_outputs, labels)

        # Combined loss
        loss = (
            ALPHA * distillation_loss
            + (1 - ALPHA) * student_loss
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    accuracy = evaluate(student)

    history.append({
        "epoch": epoch + 1,
        "loss": average_loss,
        "accuracy": accuracy
    })

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {average_loss:.4f} "
        f"Accuracy: {accuracy:.2f}%"
    )

# Final evaluation
teacher_accuracy = evaluate(teacher)
student_accuracy = evaluate(student)

student_path = RESULTS / "student_distilled.pth"
torch.save(student.state_dict(), student_path)

student_size = student_path.stat().st_size / (1024 * 1024)

teacher_params = sum(p.numel() for p in teacher.parameters())
student_params = sum(p.numel() for p in student.parameters())

with open(RESULTS / "distillation_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["model", "parameters", "accuracy", "size_mb"]
    )
    writer.writeheader()

    writer.writerow({
        "model": "Teacher",
        "parameters": teacher_params,
        "accuracy": teacher_accuracy,
        "size_mb": (
            RESULTS / "baseline_model.pth"
        ).stat().st_size / (1024 * 1024)
    })

    writer.writerow({
        "model": "Student",
        "parameters": student_params,
        "accuracy": student_accuracy,
        "size_mb": student_size
    })

with open(RESULTS / "distillation_history.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["epoch", "loss", "accuracy"]
    )
    writer.writeheader()
    writer.writerows(history)

print("\n========== DISTILLATION RESULTS ==========")
print(f"Teacher accuracy: {teacher_accuracy:.2f}%")
print(f"Student accuracy: {student_accuracy:.2f}%")
print(f"Teacher parameters: {teacher_params:,}")
print(f"Student parameters: {student_params:,}")
print(f"Student size: {student_size:.3f} MB")
print("==========================================")
