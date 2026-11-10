
import csv
import copy
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from train import (
    config,
    DEVICE,
    CNN,
    create_dataloaders,
    set_seed
)


# --------------------------------------------------
# 1. Experiment configuration
# --------------------------------------------------

EPOCHS = 10

EXPERIMENTS = [
    ("SGD", "StepLR"),
    ("SGD", "CosineAnnealingLR"),
    ("SGD", "ReduceLROnPlateau"),
    ("Adam", "StepLR"),
    ("Adam", "CosineAnnealingLR"),
    ("Adam", "ReduceLROnPlateau"),
]


# --------------------------------------------------
# 2. Create optimizer and scheduler
# --------------------------------------------------

def create_optimizer(model, optimizer_name):
    if optimizer_name == "SGD":
        return torch.optim.SGD(
            model.parameters(),
            lr=0.01,
            momentum=0.9
        )

    return torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )


def create_scheduler(optimizer, scheduler_name):
    if scheduler_name == "StepLR":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=4,
            gamma=0.5
        )

    if scheduler_name == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=EPOCHS
        )

    if scheduler_name == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2
        )

    raise ValueError(f"Unknown scheduler: {scheduler_name}")


# --------------------------------------------------
# 3. Training and evaluation
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
# 4. Run one experiment
# --------------------------------------------------

def run_experiment(optimizer_name, scheduler_name):
    print("\n" + "=" * 55)
    print(f"Optimizer: {optimizer_name}")
    print(f"Scheduler: {scheduler_name}")
    print("=" * 55)

    set_seed(config["project"]["seed"])

    train_loader, val_loader, test_loader = create_dataloaders()

    model = CNN(
        num_classes=config["model"]["num_classes"]
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = create_optimizer(model, optimizer_name)
    scheduler = create_scheduler(optimizer, scheduler_name)

    history = []
    best_val_accuracy = 0.0
    best_state = None

    start_time = time.perf_counter()

    for epoch in range(EPOCHS):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion
        )

        # Scheduler update
        if scheduler_name == "ReduceLROnPlateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "learning_rate": current_lr
        })

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Acc: {val_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"LR: {current_lr:.6f}"
        )

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    training_time = time.perf_counter() - start_time

    # Evaluate the best validation checkpoint
    model.load_state_dict(best_state)

    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        criterion
    )

    total_samples = len(train_loader.dataset) * EPOCHS
    throughput = total_samples / training_time

    result = {
        "optimizer": optimizer_name,
        "scheduler": scheduler_name,
        "training_time": training_time,
        "throughput": throughput,
        "best_val_accuracy": best_val_accuracy,
        "test_accuracy": test_accuracy,
        "test_loss": test_loss
    }

    print("\nExperiment result:")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Throughput: {throughput:.2f} samples/sec")
    print(f"Best validation accuracy: {best_val_accuracy * 100:.2f}%")
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")

    return result, history


# --------------------------------------------------
# 5. Main experiment
# --------------------------------------------------

def main():
    print("===== OPTIMIZER AND SCHEDULER STUDY =====")
    print("Device:", DEVICE)
    print("Epochs per experiment:", EPOCHS)
    print("Total experiments:", len(EXPERIMENTS))

    results = []
    histories = {}

    for optimizer_name, scheduler_name in EXPERIMENTS:
        result, history = run_experiment(
            optimizer_name,
            scheduler_name
        )

        results.append(result)

        experiment_name = f"{optimizer_name}_{scheduler_name}"
        histories[experiment_name] = history

    # Results directory
    project_root = Path(__file__).resolve().parents[2]
    results_dir = project_root / "lab06" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save summary CSV
    csv_path = results_dir / "scheduler_study.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys())
        )
        writer.writeheader()
        writer.writerows(results)

    # Save epoch-by-epoch history
    history_path = results_dir / "scheduler_training_history.csv"

    with open(history_path, "w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "experiment",
            "epoch",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
            "learning_rate"
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for name, history in histories.items():
            for row in history:
                writer.writerow({
                    "experiment": name,
                    **row
                })

    # Plot validation accuracy
    plt.figure(figsize=(10, 6))

    for name, history in histories.items():
        epochs = [row["epoch"] for row in history]
        accuracy = [
            row["val_accuracy"] * 100
            for row in history
        ]

        plt.plot(epochs, accuracy, marker="o", label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Validation accuracy (%)")
    plt.title("Optimizer and Scheduler Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    accuracy_plot = results_dir / "scheduler_accuracy.png"
    plt.savefig(accuracy_plot, dpi=150)
    plt.close()

    # Plot throughput
    plt.figure(figsize=(10, 6))

    names = [
        f"{r['optimizer']}\n{r['scheduler']}"
        for r in results
    ]

    throughputs = [r["throughput"] for r in results]

    plt.bar(names, throughputs)
    plt.ylabel("Throughput (samples/sec)")
    plt.title("Optimizer and Scheduler Throughput")
    plt.grid(axis="y")
    plt.tight_layout()

    throughput_plot = results_dir / "scheduler_throughput.png"
    plt.savefig(throughput_plot, dpi=150)
    plt.close()

    # Print final comparison
    print("\n===== FINAL COMPARISON =====")

    for result in results:
        print(
            f"{result['optimizer']} + {result['scheduler']} | "
            f"Time: {result['training_time']:.2f}s | "
            f"Throughput: {result['throughput']:.2f} | "
            f"Test Accuracy: {result['test_accuracy'] * 100:.2f}%"
        )

    best = max(results, key=lambda x: x["test_accuracy"])

    print("\nBest configuration:")
    print(f"Optimizer: {best['optimizer']}")
    print(f"Scheduler: {best['scheduler']}")
    print(f"Test accuracy: {best['test_accuracy'] * 100:.2f}%")

    print("\nSaved files:")
    print(csv_path)
    print(history_path)
    print(accuracy_plot)
    print(throughput_plot)

    print("\nScheduler experiment completed!")


if __name__ == "__main__":
    main()
