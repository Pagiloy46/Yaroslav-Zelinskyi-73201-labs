
from pathlib import Path

from sklearn.datasets import fetch_openml


def main():
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data"
    results_dir = project_root / "lab04" / "results"

    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading UCI Adult dataset...")

    adult = fetch_openml(
        name="adult",
        version=2,
        as_frame=True,
        data_home=str(data_dir / "openml"),
    )

    X = adult.data
    y = adult.target

    print("\nDataset loaded successfully!")
    print("Rows:", len(X))
    print("Feature columns:", X.shape[1])
    print("Target name:", y.name)
    print("Total columns including target:", X.shape[1] + 1)

    # Save an unchanged copy of the raw data.
    raw_data = X.copy()
    raw_data[y.name] = y

    raw_path = data_dir / "adult_raw.csv"
    raw_data.to_csv(raw_path, index=False)

    # Record ingestion metadata.
    metadata_path = results_dir / "ingestion_metadata.txt"
    with metadata_path.open("w", encoding="utf-8") as file:
        file.write("Dataset: UCI Adult via OpenML\n")
        file.write("Name: adult\n")
        file.write("Version: 2\n")
        file.write(f"Rows: {len(raw_data)}\n")
        file.write(f"Feature columns: {X.shape[1]}\n")
        file.write(f"Columns including target: {raw_data.shape[1]}\n")
        file.write(f"Target column: {y.name}\n")

    print("Raw dataset saved to:", raw_path)
    print("Metadata saved to:", metadata_path)


if __name__ == "__main__":
    main()
