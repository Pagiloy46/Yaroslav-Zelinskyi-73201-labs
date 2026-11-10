
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset

from train import (
    config,
    DEVICE,
    CNN,
    create_dataloaders,
    set_seed
)


TOTAL_EPOCHS = 5
INTERRUPT_EPOCH = 3
BATCH_SIZE = 128


def save_checkpoint(path, epoch, model, optimizer, history):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "seed": config["project"]["seed"]
    }

    torch.save(checkpoint, path)
    print(f"Checkpoint saved after epoch {epoch}")


def load_checkpoint(path, model, optimizer):
    checkpoint = torch.load(
        path,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"Resuming from epoch {checkpoint['epoch'] + 1}")

    return checkpoint["epoch"], checkpoint["history"]


def train_epoch(model, loader, criterion, optimizer):
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


def main():
    print("===== CHECKPOINT AND RESUME EXPERIMENT =====")

    set_seed(config["project"]["seed"])

    train_loader_original, val_loader, _ = create_dataloaders()

    # Smaller dataset for a faster experiment
    train_dataset = Subset(
        train_loader_original.dataset,
        range(5000)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
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

    root = Path(__file__).resolve().parents[2]
    results_dir = root / "lab06" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = results_dir / "resume_checkpoint.pth"

    history = []
    start_epoch = 0

    # Phase 1: train until epoch 3
    print("\n===== PHASE 1: INITIAL TRAINING =====")

    for epoch in range(INTERRUPT_EPOCH):
        loss, accuracy = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        history.append({
            "epoch": epoch + 1,
            "loss": loss,
            "accuracy": accuracy
        })

        print(
            f"Epoch {epoch + 1}/{TOTAL_EPOCHS} | "
            f"Loss: {loss:.4f} | "
            f"Accuracy: {accuracy * 100:.2f}%"
        )

    save_checkpoint(
        checkpoint_path,
        INTERRUPT_EPOCH,
        model,
        optimizer,
        history
    )

    print("\nTraining interrupted after epoch 3.")

    # Phase 2: restore model and optimizer
    print("\n===== PHASE 2: RESUME TRAINING =====")

    restored_model = CNN(
        num_classes=config["model"]["num_classes"]
    ).to(DEVICE)

    restored_optimizer = torch.optim.Adam(
        restored_model.parameters(),
        lr=config["training"]["learning_rate"]
    )

    start_epoch, history = load_checkpoint(
        checkpoint_path,
        restored_model,
        restored_optimizer
    )

    for epoch in range(start_epoch, TOTAL_EPOCHS):
        loss, accuracy = train_epoch(
            restored_model,
            train_loader,
            criterion,
            restored_optimizer
        )

        history.append({
            "epoch": epoch + 1,
            "loss": loss,
            "accuracy": accuracy
        })

        print(
            f"Epoch {epoch + 1}/{TOTAL_EPOCHS} | "
            f"Loss: {loss:.4f} | "
            f"Accuracy: {accuracy * 100:.2f}%"
        )

    # Save final model
    final_path = results_dir / "resumed_model.pth"

    torch.save(
        restored_model.state_dict(),
        final_path
    )

    # Save history
    history_path = results_dir / "resume_training_history.csv"

    with open(history_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["epoch", "loss", "accuracy"]
        )

        writer.writeheader()
        writer.writerows(history)

    print("\n===== FINAL RESULTS =====")
    print("Completed epochs:", len(history))
    print("Checkpoint:", checkpoint_path)
    print("Final model:", final_path)
    print("Training history:", history_path)

    print("\nCheckpoint and resume experiment completed!")


if __name__ == "__main__":
    main()
