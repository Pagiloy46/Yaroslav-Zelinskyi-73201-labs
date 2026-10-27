
from pathlib import Path

import pandas as pd


EXPECTED_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "class",
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
    "class",
]

NUMERIC_COLUMNS = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

RANGES = {
    "age": (17, 90),
    "hours-per-week": (1, 99),
    "capital-gain": (0, 99999),
    "capital-loss": (0, 4356),
}


def main():
    project_root = Path(__file__).resolve().parents[2]
    raw_path = project_root / "data" / "adult_raw.csv"
    results_dir = project_root / "lab04" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_path)

    violations = []

    # Schema checks
    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
    extra_columns = sorted(set(df.columns) - set(EXPECTED_COLUMNS))

    if missing_columns:
        violations.append(f"Missing columns: {missing_columns}")
    if extra_columns:
        violations.append(f"Unexpected columns: {extra_columns}")

    # Numeric type checks
    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            converted = pd.to_numeric(df[column], errors="coerce")
            invalid = converted.isna() & df[column].notna()
            count = int(invalid.sum())
            if count:
                violations.append(
                    f"{column}: {count} values cannot be parsed as numeric"
                )

    # Missing-value checks: Adult often represents missing values as '?'
    question_marks = df.astype("string").apply(
        lambda col: col.str.strip().eq("?")
    )

    for column in df.columns:
        count = int(question_marks[column].sum())
        if count:
            violations.append(
                f"{column}: {count} missing-value markers '?'"
            )

    # Range checks
    for column, (minimum, maximum) in RANGES.items():
        if column in df.columns:
            values = pd.to_numeric(df[column], errors="coerce")
            invalid = values.notna() & (
                (values < minimum) | (values > maximum)
            )
            count = int(invalid.sum())
            if count:
                violations.append(
                    f"{column}: {count} values outside [{minimum}, {maximum}]"
                )

    # Duplicate and label-conflict checks
    feature_columns = [c for c in df.columns if c != "class"]

    duplicate_features = df.duplicated(
        subset=feature_columns, keep=False
    )
    duplicate_count = int(duplicate_features.sum())

    conflict_count = 0
    if duplicate_count:
        duplicate_rows = df.loc[duplicate_features]
        label_counts = duplicate_rows.groupby(
            feature_columns, dropna=False
        )["class"].nunique(dropna=False)
        conflict_count = int((label_counts > 1).sum())

    violations.append(
        f"Duplicate feature rows (including repeated occurrences): "
        f"{duplicate_count}"
    )
    violations.append(
        f"Duplicate feature groups with conflicting labels: {conflict_count}"
    )

    # Save validation report without silently dropping or modifying rows.
    report_path = results_dir / "validation_report.txt"
    with report_path.open("w", encoding="utf-8") as file:
        file.write(f"Rows: {len(df)}\n")
        file.write(f"Columns: {len(df.columns)}\n")
        file.write(f"Column names: {list(df.columns)}\n\n")
        file.write("Validation findings:\n")

        if violations:
            for item in violations:
                file.write(f"- {item}\n")
        else:
            file.write("- No schema, type, missing-marker, or range violations.\n")

    print("Validation completed.")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Duplicate feature rows:", duplicate_count)
    print("Conflicting-label groups:", conflict_count)
    print("Findings:", len(violations))
    print("Report saved to:", report_path)

    print("\nValidation findings:")
    for item in violations:
        print("-", item)


if __name__ == "__main__":
    main()
