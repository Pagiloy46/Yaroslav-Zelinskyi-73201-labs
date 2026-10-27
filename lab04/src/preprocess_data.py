
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


SEED = 42
TARGET = "class"

NUMERIC_COLUMNS = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

CATEGORICAL_COLUMNS = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

RANGE_COLUMNS = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]


def clean_missing_markers(df):
    """Replace '?' markers with missing values."""
    df = df.copy()

    for column in df.select_dtypes(include=["object", "string"]).columns:
        df[column] = df[column].replace(r"^\s*\?\s*$", np.nan, regex=True)
        df[column] = df[column].str.strip()

    return df


def add_derived_features(df):
    """Create the two required derived features."""
    df = df.copy()

    df["capital_net"] = df["capital-gain"] - df["capital-loss"]
    df["hours_per_age"] = (
        df["hours-per-week"] / (df["age"] - 16)
    )

    return df


def clip_numeric(df, lower_bounds, upper_bounds):
    """Clip numeric features using training-set quantiles."""
    df = df.copy()

    for column in RANGE_COLUMNS:
        df[column] = df[column].clip(
            lower=lower_bounds[column],
            upper=upper_bounds[column],
        )

    return df


def make_preprocessor(numeric_features):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
    ])


def make_model():
    return LogisticRegression(
        max_iter=5000,
        solver="lbfgs",
        random_state=SEED,
    )


def calculate_metrics(y_true, predictions, variant):
    # The positive class is income >50K.
    positive_label = ">50K"

    # Support both common Adult target formats.
    if positive_label not in set(y_true):
        positive_label = ">50K."

    return {
        "variant": variant,
        "accuracy": accuracy_score(y_true, predictions),
        "f1": f1_score(
            y_true,
            predictions,
            pos_label=positive_label,
            zero_division=0,
        ),
    }


def main():
    project_root = Path(__file__).resolve().parents[2]
    raw_path = project_root / "data" / "adult_raw.csv"
    results_dir = project_root / "lab04" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load raw data without modifying the source file.
    df = pd.read_csv(raw_path)
    df = clean_missing_markers(df)

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Report rows with missing target labels instead of silently dropping them.
    missing_target_count = int(df[TARGET].isna().sum())
    print("Rows with missing target:", missing_target_count)

    if missing_target_count:
        missing_report = results_dir / "missing_target_rows.txt"
        missing_report.write_text(
            f"Rows with missing target excluded from modeling: "
            f"{missing_target_count}\n",
            encoding="utf-8",
        )
        df = df.loc[df[TARGET].notna()].copy()

    # Normalize target formatting, e.g. '>50K.' -> '>50K'.
    df[TARGET] = df[TARGET].astype(str).str.strip().str.rstrip(".")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Split before fitting imputers, scalers, encoders, or clipping limits.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=SEED,
        stratify=y,
    )

    # Fit outlier boundaries on training data only.
    lower_bounds = X_train[RANGE_COLUMNS].quantile(0.01).to_dict()
    upper_bounds = X_train[RANGE_COLUMNS].quantile(0.99).to_dict()

    X_train = clip_numeric(X_train, lower_bounds, upper_bounds)
    X_test = clip_numeric(X_test, lower_bounds, upper_bounds)

    # Engineer derived features after clipping.
    X_train = add_derived_features(X_train)
    X_test = add_derived_features(X_test)

    derived_numeric_features = NUMERIC_COLUMNS + [
        "capital_net",
        "hours_per_age",
    ]

    # -------------------------------
    # Model WITH derived features
    # -------------------------------
    derived_preprocessor = make_preprocessor(
        derived_numeric_features
    )

    X_train_derived = derived_preprocessor.fit_transform(X_train)
    X_test_derived = derived_preprocessor.transform(X_test)

    derived_model = make_model()
    derived_model.fit(X_train_derived, y_train)
    derived_predictions = derived_model.predict(X_test_derived)

    # -------------------------------
    # Model WITHOUT derived features
    # -------------------------------
    baseline_preprocessor = make_preprocessor(NUMERIC_COLUMNS)

    X_train_baseline = baseline_preprocessor.fit_transform(
        X_train[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS]
    )
    X_test_baseline = baseline_preprocessor.transform(
        X_test[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS]
    )

    baseline_model = make_model()
    baseline_model.fit(X_train_baseline, y_train)
    baseline_predictions = baseline_model.predict(X_test_baseline)

    metrics = [
        calculate_metrics(
            y_test,
            baseline_predictions,
            "without_derived_features",
        ),
        calculate_metrics(
            y_test,
            derived_predictions,
            "with_derived_features",
        ),
    ]

    metrics_path = results_dir / "baseline_metrics.csv"
    pd.DataFrame(metrics).to_csv(metrics_path, index=False)

    # Save the fitted preprocessor that includes the derived features.
    joblib.dump(
        derived_preprocessor,
        results_dir / "preprocessor.joblib",
    )

    # Save processed training data and target.
    train_array = (
        X_train_derived.toarray()
        if hasattr(X_train_derived, "toarray")
        else np.asarray(X_train_derived)
    )

    processed_df = pd.DataFrame(train_array)
    processed_df[TARGET] = y_train.to_numpy()

    processed_path = results_dir / "processed_train.csv"
    processed_df.to_csv(processed_path, index=False)

    print("\nPreprocessing completed.")
    print("Training rows:", len(X_train))
    print("Test rows:", len(X_test))
    print("Processed train shape:", X_train_derived.shape)
    print("Processed test shape:", X_test_derived.shape)

    print("\nMetrics:")
    print(pd.DataFrame(metrics).to_string(index=False))

    print("\nSaved:", metrics_path)
    print("Saved:", results_dir / "preprocessor.joblib")
    print("Saved:", processed_path)


if __name__ == "__main__":
    main()
