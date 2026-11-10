
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, record_function

from train import (
    config,
    DEVICE,
    CNN,
    create_dataloaders,
    set_seed
)


def synchronize():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def main():
    print("===== PYTORCH PROFILER =====")
    print("Device:", DEVICE)

    set_seed(config["project"]["seed"])

    # Load data
    train_loader, _, _ = create_dataloaders()

    # Create model
    model = CNN(
        num_classes=config["model"]["num_classes"]
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"]
    )

    model.train()

    # Time measurements
    times = {
        "Data loading": 0.0,
        "Forward pass": 0.0,
        "Backward pass": 0.0,
        "Optimizer step": 0.0
    }

    total_batches = len(train_loader)

    activities = [ProfilerActivity.CPU]

    if DEVICE.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    print("\nProfiling one training epoch...")
    print("Number of batches:", total_batches)

    # Profile one epoch
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False
    ) as prof:

        iterator = iter(train_loader)

        for batch_idx in range(total_batches):

            # 1. Data loading
            start = time.perf_counter()

            with record_function("Data Loading"):
                images, labels = next(iterator)
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

            synchronize()
            times["Data loading"] += time.perf_counter() - start

            # 2. Forward pass
            optimizer.zero_grad()

            start = time.perf_counter()

            with record_function("Forward Pass"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            synchronize()
            times["Forward pass"] += time.perf_counter() - start

            # 3. Backward pass
            start = time.perf_counter()

            with record_function("Backward Pass"):
                loss.backward()

            synchronize()
            times["Backward pass"] += time.perf_counter() - start

            # 4. Optimizer step
            start = time.perf_counter()

            with record_function("Optimizer Step"):
                optimizer.step()

            synchronize()
            times["Optimizer step"] += time.perf_counter() - start

            prof.step()

            if (batch_idx + 1) % 100 == 0:
                print(
                    f"Processed {batch_idx + 1}/{total_batches} batches"
                )

    # Results directory
    project_root = Path(__file__).resolve().parents[2]
    results_dir = project_root / "lab06" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Calculate total time and percentages
    total_time = sum(times.values())

    print("\n===== PROFILING BREAKDOWN =====")

    for name, duration in times.items():
        percentage = duration / total_time * 100

        print(
            f"{name}: {duration:.4f} seconds "
            f"({percentage:.2f}%)"
        )

    print(f"\nTotal measured time: {total_time:.4f} seconds")

    # Save breakdown to CSV
    csv_path = results_dir / "profiling_breakdown.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow(["Stage", "Time (seconds)", "Percentage"])

        for name, duration in times.items():
            percentage = duration / total_time * 100
            writer.writerow([
                name,
                round(duration, 6),
                round(percentage, 2)
            ])

    # Save Chrome trace
    trace_path = results_dir / "profiler_trace.json"
    prof.export_chrome_trace(str(trace_path))

    # Save PyTorch operator table
    table_path = results_dir / "profiler_operator_table.txt"

    table = prof.key_averages().table(
        sort_by="self_cpu_time_total",
        row_limit=20
    )

    with open(table_path, "w", encoding="utf-8") as file:
        file.write(table)

    print("\n===== TOP PYTORCH OPERATORS =====")
    print(table)

    print("\n===== SAVED FILES =====")
    print("Breakdown:", csv_path)
    print("Chrome trace:", trace_path)
    print("Operator table:", table_path)

    print("\nProfiling completed!")


if __name__ == "__main__":
    main()
