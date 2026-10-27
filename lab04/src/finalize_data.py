
from pathlib import Path
import hashlib
import json

import pandas as pd
import yaml


SEED = 42


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    root = Path(__file__).resolve().parents[2]
    results = root / "lab04" / "results"

    processed_path = results / "processed_train.csv"
    metrics_path = results / "baseline_metrics.csv"
    preprocessor_path = results / "preprocessor.joblib"

    for path in (processed_path, metrics_path, preprocessor_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}. "
                "Run preprocess_data.py first."
            )

    # Hash the final processed training data.
    digest = sha256_file(processed_path)

    (results / "processed_data.sha256").write_text(
        f"{digest}  {processed_path.name}\n",
        encoding="utf-8",
    )

    processed = pd.read_csv(processed_path)
    metrics = pd.read_csv(metrics_path)

    config = {
        "dataset": {
            "name": "UCI Adult",
            "source": "OpenML",
            "openml_name": "adult",
            "version": 2,
            "raw_rows": 48842,
            "raw_columns_including_target": 15,
        },
        "reproducibility": {
            "random_seed": SEED,
            "python_split": "train_test_split",
            "test_size": 0.30,
            "stratify": True,
        },
        "missing_values": {
            "marker": "?",
            "numeric_imputation": "median",
            "categorical_imputation": "most_frequent",
        },
        "outliers": {
            "method": "quantile clipping",
            "lower_quantile": 0.01,
            "upper_quantile": 0.99,
            "fit_on": "training data only",
        },
        "numeric_preprocessing": {
            "scaling": "StandardScaler",
        },
        "categorical_preprocessing": {
            "encoding": "OneHotEncoder",
            "handle_unknown": "ignore",
        },
        "derived_features": [
            "capital_net = capital-gain - capital-loss",
            "hours_per_age = hours-per-week / (age - 16)",
        ],
        "model": {
            "name": "LogisticRegression",
            "solver": "lbfgs",
            "max_iter": 5000,
            "random_state": SEED,
        },
        "processed_data": {
            "file": processed_path.name,
            "rows": int(processed.shape[0]),
            "columns_including_target": int(processed.shape[1]),
            "sha256": digest,
        },
    }

    config_path = results / "config.yaml"
    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            config,
            file,
            sort_keys=False,
            allow_unicode=True,
        )

    metadata = {
        "file": processed_path.name,
        "rows": int(processed.shape[0]),
        "columns_including_target": int(processed.shape[1]),
        "sha256": digest,
        "metrics": metrics.to_dict(orient="records"),
    }

    metadata_path = results / "processed_data_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("Finalization completed.")
    print("Processed rows:", processed.shape[0])
    print("Columns including target:", processed.shape[1])
    print("SHA-256:", digest)
    print("Metrics:")
    print(metrics.to_string(index=False))
    print("Updated:", results / "processed_data.sha256")
    print("Updated:", config_path)
    print("Updated:", metadata_path)


if __name__ == "__main__":
    main()
