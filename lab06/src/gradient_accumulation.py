
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset
from train import config, DEVICE, CNN, create_dataloaders, set_seed


# --------------------------------------------------
# 1. Experiment settings
# --------------------------------------------------

EPOCHS = 5
BASE_BATCH_SIZE = 256
MICRO_BATCH_SIZE = 64
ACCUMULATION_STEPS = 4
STUDY_SIZE = 5120


# --------------------------------------------------
# 2. Training without accumulation
# --------------------------------------------------

def train_normal(model, loader, optimizer, criterion):
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
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


# --------------------------------------------------
# 3. Training with gradient accumulation
# --------------------------------------------------

def train_accumulated(model, loader, optimizer, criterion):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    optimizer.zero_grad()

    for batch_index, (images, labels) in enumerate(loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        # Divide loss to average gradients over 4 mini-batches
        (loss / ACCUMULATION_STEPS).backward()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

        # Update weights after 4 mini-batches
        if (batch_index + 1) % ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()

    return total_loss / total, correct / total


# --------------------------------------------------
# 4. Run one experiment
# --------------------------------------------------

def run_experiment(use_accumulation, train_dataset):
    set_seed(config["project"]["seed"])

    batch_size = (
        MICRO_BATCH_SIZE if use_accumulation
        else BASE_BATCH_SIZE
    )

    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(
            config["project"]["seed"]
        )
    )

    model = CNN(
        num_classes=config["model"]["num_classes"]
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"]
    )

    criterion = nn.CrossEntropyLoss()

    history = []

    start_time = time.perf_counter()

    for epoch in range(EPOCHS):
        if use_accumulation:
            loss, accuracy = train_accumulated(
                model, loader, optimizer, criterion
            )
        else:
            loss, accuracy = train_normal(
                model, loader, optimizer, criterion
            )

        history.append({
            "epoch": epoch + 1,
            "loss": loss,
            "accuracy": accuracy
        })

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Loss: {loss:.4f} | "
            f"Accuracy: {accuracy * 100:.2f}%"
        )

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start_time

    return {
        "mode": (
            "Gradient Accumulation"
            if use_accumulation else "Normal Batch 256"
        ),
        "batch_size": batch_size,
        "effective_batch_size": BASE_BATCH_SIZE,
        "training_time": round(elapsed, 2),
        "final_loss": round(history[-1]["loss"], 4),
        "final_accuracy": round(history[-1]["accuracy"], 4)
    }, history


# --------------------------------------------------
# 5. Main experiment
# --------------------------------------------------

def main():
    print("===== GRADIENT ACCUMULATION EXPERIMENT =====")
    print("Device:", DEVICE)
    print("Normal batch size:", BASE_BATCH_SIZE)
    print("Micro batch size:", MICRO_BATCH_SIZE)
    print("Accumulation steps:", ACCUMULATION_STEPS)
    print("Effective batch size:", BASE_BATCH_SIZE)
    print("Epochs:", EPOCHS)

    train_loader, _, _ = create_dataloaders()

    train_dataset = Subset(
        train_loader.dataset,
        range(STUDY_SIZE)
    )

    results = []
    histories = {}

    print("\n--- Normal batch training ---")

    normal_result, normal_history = run_experiment(
        False, train_dataset
    )

    results.append(normal_result)
    histories["Normal Batch 256"] = normal_history

    print("\n--- Gradient accumulation training ---")

    accumulation_result, accumulation_history = run_experiment(
        True, train_dataset
    )

    results.append(accumulation_result)
    histories["Gradient Accumulation"] = accumulation_history

    # Save results
    root = Path(__file__).resolve().parents[2]
    results_dir = root / "lab06" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "gradient_accumulation.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys())
        )
        writer.writeheader()
        writer.writerows(results)

    # Save epoch history
    history_path = results_dir / "gradient_accumulation_history.csv"

    with open(history_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            "mode", "epoch", "loss", "accuracy"
        ])

        for name, history in histories.items():
            for row in history:
                writer.writerow([
                    name,
                    row["epoch"],
                    row["loss"],
                    row["accuracy"]
                ])

    # Plot accuracy
    plt.figure(figsize=(8, 5))

    for name, history in histories.items():
        plt.plot(
            [x["epoch"] for x in history],
            [x["accuracy"] * 100 for x in history],
            marker="o",
            label=name
        )

    plt.xlabel("Epoch")
    plt.ylabel("Training accuracy (%)")
    plt.title("Normal Batch vs Gradient Accumulation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_path = results_dir / "gradient_accumulation.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print("\n===== FINAL COMPARISON =====")

    for result in results:
        print(
            f"{result['mode']} | "
            f"Time: {result['training_time']}s | "
            f"Loss: {result['final_loss']} | "
            f"Accuracy: {result['final_accuracy'] * 100:.2f}%"
        )

    print("\nSaved results:")
    print(csv_path)
    print(history_path)
    print(plot_path)

    print("\nGradient accumulation experiment completed!")


if __name__ == "__main__":
    main()
