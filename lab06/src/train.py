
import os
import csv
import random
import yaml
import mlflow
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


# --------------------------------------------------
# 1. Load configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = PROJECT_ROOT / "lab06" / "configs" / "config.yaml"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


config = load_config()


# --------------------------------------------------
# 2. Set random seed
# --------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SEED = config["project"]["seed"]
set_seed(SEED)


# --------------------------------------------------
# 3. Select device
# --------------------------------------------------

if config["project"]["device"] == "auto":
    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
else:
    DEVICE = torch.device(config["project"]["device"])

print("Using device:", DEVICE)


# --------------------------------------------------
# 4. CNN model from Lab 3
# --------------------------------------------------

class CNN(nn.Module):
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
            nn.Linear(32 * 7 * 7, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# --------------------------------------------------
# 5. Load Fashion-MNIST
# --------------------------------------------------

def create_dataloaders():
    data_config = config["data"]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    data_root = PROJECT_ROOT / data_config["root"]

    train_dataset = datasets.FashionMNIST(
        root=str(data_root),
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.FashionMNIST(
        root=str(data_root),
        train=False,
        download=True,
        transform=transform
    )

    train_size = data_config["train_size"]
    val_size = data_config["val_size"]

    train_subset, val_subset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    batch_size = data_config["batch_size"]
    num_workers = data_config["num_workers"]

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(DEVICE.type == "cuda")
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(DEVICE.type == "cuda")
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(DEVICE.type == "cuda")
    )

    return train_loader, val_loader, test_loader


# --------------------------------------------------
# 6. Training function
# --------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


# --------------------------------------------------
# 7. Evaluation function
# --------------------------------------------------

def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


# --------------------------------------------------
# 8. Main training pipeline
# --------------------------------------------------

def main():
    train_loader, val_loader, test_loader = create_dataloaders()

    model = CNN(
        num_classes=config["model"]["num_classes"]
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"]
    )

    scheduler_config = config["training"]

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=scheduler_config["step_size"],
        gamma=scheduler_config["gamma"]
    )

    checkpoint_dir = (
        PROJECT_ROOT / scheduler_config["checkpoint_dir"]
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    results_dir = PROJECT_ROOT / "lab06" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    best_accuracy = 0.0
    history = []

    # MLflow configuration
    mlflow.set_tracking_uri(
        config["logging"]["tracking_uri"]
    )

    mlflow.set_experiment(
        config["logging"]["mlflow_experiment"]
    )

    with mlflow.start_run(run_name="CNN_training_pipeline"):

        mlflow.log_params({
            "seed": SEED,
            "batch_size": config["data"]["batch_size"],
            "num_workers": config["data"]["num_workers"],
            "epochs": scheduler_config["epochs"],
            "learning_rate": scheduler_config["learning_rate"],
            "optimizer": scheduler_config["optimizer"],
            "scheduler": scheduler_config["scheduler"],
            "model": config["model"]["name"]
        })

        mlflow.log_param(
            "parameters",
            sum(p.numel() for p in model.parameters())
        )

        print("\nStarting training...\n")

        for epoch in range(scheduler_config["epochs"]):

            train_loss, train_accuracy = train_one_epoch(
                model, train_loader, criterion, optimizer
            )

            val_loss, val_accuracy = evaluate(
                model, val_loader, criterion
            )

            scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch [{epoch + 1}/"
                f"{scheduler_config['epochs']}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_accuracy:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_accuracy:.4f}"
            )

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "learning_rate": current_lr
            }, step=epoch + 1)

            history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "learning_rate": current_lr
            })

            # Save latest checkpoint
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_accuracy": best_accuracy,
                "seed": SEED
            }

            torch.save(
                checkpoint,
                checkpoint_dir / "latest_checkpoint.pth"
            )

            # Save best model
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy

                torch.save(
                    checkpoint,
                    checkpoint_dir / "best_model.pth"
                )

        # Test best model
        best_checkpoint = torch.load(
            checkpoint_dir / "best_model.pth",
            map_location=DEVICE,
            weights_only=False
        )

        model.load_state_dict(
            best_checkpoint["model_state_dict"]
        )

        test_loss, test_accuracy = evaluate(
            model, test_loader, criterion
        )

        print("\n===== FINAL TEST RESULTS =====")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

        mlflow.log_metrics({
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "best_val_accuracy": best_accuracy
        })

        # Save training history
        history_path = results_dir / "training_history.csv"

        with open(history_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=history[0].keys()
            )
            writer.writeheader()
            writer.writerows(history)

        mlflow.log_artifact(str(history_path))
        mlflow.log_artifact(
            str(checkpoint_dir / "best_model.pth")
        )

        print("\nTraining completed!")
        print("Best model:", checkpoint_dir / "best_model.pth")
        print("History:", history_path)


if __name__ == "__main__":
    main()
