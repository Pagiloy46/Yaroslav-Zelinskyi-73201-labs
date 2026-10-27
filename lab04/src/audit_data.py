
from pathlib import Path

import pandas as pd


SEED = 42

NUMERIC_RANGES = {
    "age": (17, 90),
    "hours-per-week": (1, 99),
    "capital-gain": (0, 99999),
    "capital-loss": (0, 4356),
}

ALLOWED_TARGETS = {"<=50K", ">50K", "<=50K.", ">50K."}


def main():
    root = Path(__file__).resolve().parents[2]
    raw_path = root / "data" / "adult_raw.csv"
    results_dir = root / "lab04" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_path)

    # 1. Random sample of 50 rows for manual review.
    sample = df.sample(
        n=min(50, len(df)),
        random_state=SEED,
    )

    sample_path = results_dir / "manual_review_50.csv"
    sample.to_csv(sample_path, index=False)

    # 2. Find identical feature rows with conflicting income labels.
    features = [column for column in df.columns if column != "class"]

    label_counts = (
        df.groupby(features, dropna=False)["class"]
        .nunique(dropna=False)
    )

    conflicting_groups = label_counts[label_counts > 1].reset_index()
    conflicting_groups = conflicting_groups.drop(
        columns=["class"],
        errors="ignore",
    )

    # Keep all original rows belonging to conflicting feature groups.
    if len(conflicting_groups):
        conflicting_rows = df.merge(
            conflicting_groups[features],
            on=features,
            how="inner",
        ).sort_values(features)
    else:
        conflicting_rows = df.iloc[0:0].copy()

    conflict_path = results_dir / "conflicting_label_duplicates.csv"
    conflicting_rows.to_csv(conflict_path, index=False)

    # 3. Check ranges and target labels.
    findings = []

    for column, (minimum, maximum) in NUMERIC_RANGES.items():
        values = pd.to_numeric(df[column], errors="coerce")
        invalid = values.notna() & (
            (values < minimum) | (values > maximum)
        )
        count = int(invalid.sum())

        if count:
            findings.append(
                f"{column}: {count} rows outside [{minimum}, {maximum}]"
            )

    target_values = set(df["class"].dropna().astype(str).str.strip())
    unexpected_targets = sorted(target_values - ALLOWED_TARGETS)

    if unexpected_targets:
        findings.append(f"Unexpected target labels: {unexpected_targets}")

    findings_path = results_dir / "manual_audit_findings.txt"
    with findings_path.open("w", encoding="utf-8") as file:
        file.write(f"Dataset rows: {len(df)}\n")
        file.write(f"Sampled rows for manual review: {len(sample)}\n")
        file.write(
            "Conflicting feature groups: "
            f"{len(conflicting_groups)}\n"
        )
        file.write(
            "Rows involved in conflicting groups: "
            f"{len(conflicting_rows)}\n\n"
        )

        file.write("Automated checks:\n")
        if findings:
            for finding in findings:
                file.write(f"- {finding}\n")
        else:
            file.write("- No range or target-label violations found.\n")

        file.write(
            "\nManual review is required for manual_review_50.csv.\n"
        )

    print("Audit completed.")
    print("Dataset rows:", len(df))
    print("Sample saved:", sample_path)
    print("Conflicting feature groups:", len(conflicting_groups))
    print("Rows involved in conflicts:", len(conflicting_rows))
    print("Conflict details saved:", conflict_path)
    print("Findings saved:", findings_path)

    print("\nAutomated findings:")
    for finding in findings:
        print("-", finding)


if __name__ == "__main__":
    main()
