
import csv
import time
import threading
import psutil
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from train import config, DEVICE, CNN, set_seed


# --------------------------------------------------
# 1. Measure process memory
# --------------------------------------------------

class MemoryMonitor:
    def __init__(self):
        self.process = psutil.Process()
        self.peak_memory = 0
        self.running = False
        self.thread = None

    def monitor(self):
        while self.running:
            memory = self.process.memory_info().rss / (1024 ** 2)
            self.peak_memory = max(self.peak_memory, memory)
            time.sleep(0.01)

    def start(self):
        self.peak_memory = (
            self.process.memory_info().rss / (1024 ** 2)
        )
        self.running = True
        self.thread = threading.Thread(target=self.monitor)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

        memory = self.process.memory_info().rss / (1024 ** 2)
        self.peak_memory = max(self.peak_memory, memory)

        return self.peak_memory


# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

def load_datasets():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    root = Path(__file__).resolve().parents[2]
    data_root = root / config["data"]["root"]

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

    train_size = config["data"]["train_size"]
    val_size = config["data"]["val_size"]

    train_subset, _ = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(
            config["project"]["seed"]
        )
    )

    return train_subset, test_dataset


# --------------------------------------------------
# 3. Train and evaluate
# --------------------------------------------------

def run_experiment(batch_size, train_dataset, test_dataset):
    set_seed(config["project"]["seed"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(
            config["project"]["seed"]
        )
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    model = CNN(
        num_classes=config["model"]["num_classes"]
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"]
    )

    epochs = 5
    total_samples = 0

    monitor = MemoryMonitor()
    monitor.start()

    start_time = time.perf_counter()

    # Training
    for epoch in range(epochs):
        model.train()

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_samples += images.size(0)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    training_time = time.perf_counter() - start_time
    peak_memory = monitor.stop()

    throughput = total_samples / training_time

    # Test accuracy
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

    accuracy = correct / total

    return {
        "batch_size": batch_size,
        "training_time": training_time,
        "throughput": throughput,
        "peak_memory_mb": peak_memory,
        "accuracy": accuracy
    }


# --------------------------------------------------
# 4. Main experiment
# --------------------------------------------------

def main():
    print("===== BATCH SIZE EXPERIMENT =====")
    print("Device:", DEVICE)
    print("Epochs: 5")
    print("Workers: 0")

    set_seed(config["project"]["seed"])

    train_dataset, test_dataset = load_datasets()

    batch_sizes = [32, 64, 128, 256, 512]

    results = []

    for batch_size in batch_sizes:
        print(f"\nTesting batch size: {batch_size}")

        result = run_experiment(
            batch_size,
            train_dataset,
            test_dataset
        )

        results.append(result)

        print(f"Training time: {result['training_time']:.2f} s")
        print(f"Throughput: {result['throughput']:.2f} samples/sec")
        print(f"Peak memory: {result['peak_memory_mb']:.2f} MB")
        print(f"Test accuracy: {result['accuracy'] * 100:.2f}%")

    # Results directory
    root = Path(__file__).resolve().parents[2]
    results_dir = root / "lab06" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV
    csv_path = results_dir / "batch_size_study.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "batch_size",
                "training_time",
                "throughput",
                "peak_memory_mb",
                "accuracy"
            ]
        )
        writer.writeheader()
        writer.writerows(results)

    # Plot throughput
    plt.figure(figsize=(8, 5))
    plt.plot(
        [r["batch_size"] for r in results],
        [r["throughput"] for r in results],
        marker="o"
    )
    plt.xlabel("Batch size")
    plt.ylabel("Throughput (samples/sec)")
    plt.title("Batch Size vs Throughput")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(results_dir / "batch_size_throughput.png", dpi=150)
    plt.close()

    # Plot memory
    plt.figure(figsize=(8, 5))
    plt.plot(
        [r["batch_size"] for r in results],
        [r["peak_memory_mb"] for r in results],
        marker="o"
    )
    plt.xlabel("Batch size")
    plt.ylabel("Peak process memory (MB)")
    plt.title("Batch Size vs Peak Memory")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(results_dir / "batch_size_memory.png", dpi=150)
    plt.close()

    # Plot accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(
        [r["batch_size"] for r in results],
        [r["accuracy"] * 100 for r in results],
        marker="o"
    )
    plt.xlabel("Batch size")
    plt.ylabel("Test accuracy (%)")
    plt.title("Batch Size vs Test Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(results_dir / "batch_size_accuracy.png", dpi=150)
    plt.close()

    print("\n===== FINAL RESULTS =====")

    for r in results:
        print(
            f"Batch: {r['batch_size']} | "
            f"Throughput: {r['throughput']:.2f} samples/sec | "
            f"Memory: {r['peak_memory_mb']:.2f} MB | "
            f"Accuracy: {r['accuracy'] * 100:.2f}%"
        )

    print("\nSaved results to:", results_dir)
    print("\nBatch size experiment completed!")


if __name__ == "__main__":
    main()
