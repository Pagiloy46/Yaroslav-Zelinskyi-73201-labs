# Lab 4: Data Engineering and Reproducible Preprocessing

## 1. Goal

The goal of this laboratory was to build a reproducible data engineering pipeline for the UCI Adult income dataset. The pipeline covers data ingestion, validation, auditing, preprocessing, feature engineering, model evaluation, and artifact generation.

## 2. Dataset

The dataset was retrieved from OpenML using `fetch_openml(name="adult", version=2, as_frame=True)`.

- Raw rows: 48,842
- Feature columns: 14
- Target: `class`
- Total columns including target: 15
- Task: binary classification of income category

The original raw dataset was saved locally as `data/adult_raw.csv`. The raw data is not intended to be committed to the Git repository.

## 3. Data Validation and Audit

The validation script checked the expected schema, numeric types, selected numeric ranges, missing-value markers represented by `?`, duplicate feature rows, and conflicting target labels.

The initial audit reported:
- 48,842 raw rows
- 111 rows involved in repeated feature combinations
- 5 feature groups with conflicting income labels
- 10 rows involved in those conflicting groups

A reproducible sample of 50 rows was exported to `results/manual_review_50.csv` for manual inspection. Conflict details were saved to `results/conflicting_label_duplicates.csv`.

The raw dataset was preserved without silently deleting conflicting rows.

## 4. Preprocessing Method

The dataset was split into training and test subsets using a 70/30 stratified split with random seed 42.

The preprocessing pipeline includes:
- Replacing `?` markers with missing values
- Median imputation for numerical features
- Most-frequent imputation for categorical features
- Standard scaling of numerical features
- One-hot encoding of categorical features
- Clipping selected numerical features to the 1st and 99th percentiles, with limits calculated from training data only
- Feature engineering

Two derived features were created:
- `capital_net = capital-gain - capital-loss`
- `hours_per_age = hours-per-week / (age - 16)`

The fitted preprocessing transformer was saved as `results/preprocessor.joblib`.

All imputers, scaling parameters, encoders, and clipping limits were fitted using training data only. The held-out test set was transformed using the fitted training preprocessing objects.

## 5. Model Evaluation

Logistic Regression was evaluated with and without the derived features. The final values are recorded in `results/baseline_metrics.csv`.

| Variant | Accuracy | F1-score |
|---|---:|---:|
| Without derived features | See `baseline_metrics.csv` | See `baseline_metrics.csv` |
| With derived features | See `baseline_metrics.csv` | See `baseline_metrics.csv` |

The evaluation compares the two feature configurations using the same train/test split.

## 6. Reproducibility and Artifacts

The pipeline uses random seed 42. Configuration details are stored in `results/config.yaml`.

The following artifacts are produced:
- `results/ingestion_metadata.txt`
- `results/validation_report.txt`
- `results/manual_review_50.csv`
- `results/conflicting_label_duplicates.csv`
- `results/manual_audit_findings.txt`
- `results/baseline_metrics.csv`
- `results/preprocessor.joblib`
- `results/processed_train.csv`
- `results/processed_data.sha256`
- `results/processed_data_metadata.json`
- `results/config.yaml`

The SHA-256 checksum records the exact contents of the processed training CSV, allowing the artifact to be checked for unintended changes.

## 7. Data Card, Bias, and Privacy

The dataset contains demographic and employment-related attributes, including race and sex. It reflects historical US census-related data and may contain historical, sampling, and social biases.

The income target is a simplified binary category and is not a complete measure of financial well-being. The experiment does not establish fairness across demographic groups, causal relationships, or suitability for high-stakes decisions.

The dataset source and known limitations are documented in `data_card.md`. The license and redistribution terms should be verified against the exact OpenML dataset record before redistributing the data.

## 8. Conclusions

1. A reproducible data ingestion and validation workflow was implemented for the UCI Adult dataset. The pipeline records schema, range, missing-value, duplicate, and target-label findings.
2. Preprocessing includes train/test splitting, imputation, numerical scaling, categorical encoding, clipping, and derived features. Fitted preprocessing steps are learned from the training subset to reduce data leakage.
3. The Logistic Regression comparison, saved preprocessing artifact, YAML configuration, and SHA-256 checksum support reproducibility. The data card records limitations, potential bias, and privacy considerations.