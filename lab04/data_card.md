# Data Card: UCI Adult Income Dataset

## 1. Dataset Overview

- **Name:** UCI Adult / Adult Income
- **Source:** OpenML, dataset name `adult`, version 2
- **Original task:** Predict whether a person's annual income exceeds $50,000.
- **Total raw rows:** 48,842
- **Input features:** 14
- **Target column:** `class`
- **Total columns including target:** 15
- **Data format:** Tabular data with numerical and categorical features

## 2. Intended Use

The dataset is used for educational machine learning experiments involving data validation, missing-value handling, feature engineering, preprocessing, and binary classification.

It should not be used as the sole basis for decisions about individuals, employment, credit, insurance, or access to public services.

## 3. Data Source and License

The dataset was retrieved programmatically using:

`fetch_openml(name="adult", version=2, as_frame=True)`

The dataset originates from the UCI Adult Census Income dataset. The exact license and redistribution terms applicable to the retrieved OpenML version should be verified against its dataset record before redistribution.

## 4. Features

The dataset contains demographic and employment-related attributes, including:

- Age
- Workclass
- Final weight (`fnlwgt`)
- Education and education number
- Marital status
- Occupation
- Relationship
- Race
- Sex
- Capital gain and capital loss
- Hours worked per week
- Native country

The target column, `class`, represents the income category.

## 5. Data Quality

Initial validation was performed on all 48,842 rows.

The validation process checked:
- Expected column names and schema
- Numeric values and specified ranges
- Missing-value markers represented by `?`
- Duplicate feature rows
- Duplicate feature groups with conflicting target labels

The initial audit found 111 rows involved in repeated feature combinations and 5 feature groups with conflicting income labels. The conflict audit identified 10 rows belonging to those groups.

These findings were recorded rather than silently removed from the raw dataset.

A deterministic sample of 50 rows was exported for manual review to `results/manual_review_50.csv`.

## 6. Preprocessing

The dataset was split into training and test subsets using a 70/30 stratified split with random seed 42.

Preprocessing includes:
- Replacing `?` markers with missing values
- Median imputation for numerical features
- Most-frequent imputation for categorical features
- One-hot encoding of categorical features
- Clipping numerical features to the 1st and 99th percentiles, with bounds fitted on the training subset
- Creating `capital_net = capital-gain - capital-loss`
- Creating `hours_per_age = hours-per-week / (age - 16)`

The fitted preprocessing transformer was saved as `results/preprocessor.joblib`.

## 7. Model and Evaluation

A Logistic Regression baseline was evaluated with and without the two derived features.

| Variant | Accuracy | F1-score |
|---|---:|---:|
| Without derived features | 0.84597 | 0.63708 |
| With derived features | 0.85491 | 0.66254 |

The evaluation used the held-out test subset. These results apply to this experiment and do not establish suitability for high-stakes individual decisions.

## 8. Reproducibility

- Random seed: 42
- Dataset source: OpenML `adult`, version 2
- Split: 70% training, 30% test
- Processed training rows: 34,189
- Processed training columns, including target: 108
- Processed training data SHA-256: `8d39a988093816b955f93e79381b3d2e8adbee6b460037b2cd9eae97e3b2c5e4`

The hash is recorded in `results/processed_data.sha256`. Pipeline settings are recorded in `results/config.yaml`.

## 9. Known Biases and Limitations

The dataset reflects historical US census-related records and may contain historical, social, and sampling biases. Its demographic and employment attributes can be associated with sensitive characteristics.

The income label is a simplified binary target and does not represent a complete measure of financial well-being. Results may not generalize to other populations, time periods, or countries.

Model performance can vary across demographic groups. This experiment did not establish fairness across groups, causal relationships, or suitability for consequential decisions.

## 10. Privacy Considerations

The dataset contains demographic and employment-related information, including race and sex. Although it is a commonly used public benchmark, it should still be handled responsibly.

Do not attempt to identify individuals, combine the data with external personal records, or use model predictions to make consequential decisions about real people.

## 11. Files Produced

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