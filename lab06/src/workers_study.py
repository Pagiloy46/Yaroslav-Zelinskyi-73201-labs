
import csv
import time
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from train import config, set_seed


def main():
    print("===== NUM_WORKERS EXPERIMENT =====")

    set_seed(config["project"]["seed"])

    project_root = Path(__file__).resolve().parents[2]

    results_dir = project_root / "lab06" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Same Fashion-MNIST training data as Lab 6 pipeline
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    dataset = datasets.FashionMNIST(
        root=str(project_root / config["data"]["root"]),
        train=True,
        download=True,
        transform=transform
    )

    train_size = config["data"]["train_size"]
    dataset = Subset(dataset, range(train_size))

    batch_size = config["data"]["batch_size"]
    worker_values = [0, 1, 2, 4, 8]

    results = []

    for workers in worker_values:
        print(f"\nTesting num_workers = {workers}")

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=False
        )

        # Measure complete data-loading time
        start = time.perf_counter()

        total_samples = 0
        first_batch_time = None

        for images, labels in loader:
            if first_batch_time is None:
                first_batch_time = time.perf_counter() - start

            total_samples += images.size(0)

        total_time = time.perf_counter() - start

        throughput = total_samples / total_time

        results.append({
            "num_workers": workers,
            "total_time": total_time,
            "first_batch_time": first_batch_time,
            "throughput": throughput
        })

        print(f"Total time: {total_time:.3f} seconds")
        print(f"First batch time: {first_batch_time:.3f} seconds")
        print(f"Throughput: {throughput:.2f} samples/sec")

        del loader

    # Save results to CSV
    csv_path = results_dir / "workers_study.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "num_workers",
                "total_time",
                "first_batch_time",
                "throughput"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    # Find best worker configuration
    best = max(results, key=lambda x: x["throughput"])

    print("\n===== FINAL RESULTS =====")

    for result in results:
        print(
            f"Workers: {result['num_workers']} | "
            f"Time: {result['total_time']:.3f}s | "
            f"Throughput: {result['throughput']:.2f} samples/sec"
        )

    print("\nBest num_workers:", best["num_workers"])
    print(f"Best throughput: {best['throughput']:.2f} samples/sec")

    # Plot throughput
    workers = [r["num_workers"] for r in results]
    throughputs = [r["throughput"] for r in results]

    plt.figure(figsize=(8, 5))
    plt.plot(workers, throughputs, marker="o")
    plt.xlabel("Number of workers")
    plt.ylabel("Throughput (samples/sec)")
    plt.title("Data Loading Throughput vs num_workers")
    plt.xticks(worker_values)
    plt.grid(True)
    plt.tight_layout()

    plot_path = results_dir / "workers_throughput.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    # Plot total loading time
    times = [r["total_time"] for r in results]

    plt.figure(figsize=(8, 5))
    plt.plot(workers, times, marker="o")
    plt.xlabel("Number of workers")
    plt.ylabel("Total loading time (seconds)")
    plt.title("Data Loading Time vs num_workers")
    plt.xticks(worker_values)
    plt.grid(True)
    plt.tight_layout()

    time_plot_path = results_dir / "workers_time.png"
    plt.savefig(time_plot_path, dpi=150)
    plt.close()

    print("\nSaved files:")
    print(csv_path)
    print(plot_path)
    print(time_plot_path)

    print("\nWorkers experiment completed!")


if __name__ == "__main__":
    main()
