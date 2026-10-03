# PlacementPredict

A binary classification project predicting student placement outcomes
(`PlacementStatus`: 0 = not placed, 1 = placed) from academic performance,
test scores, and activity/experience indicators.

## Current status: pipeline complete through PCA. Classifier models are NOT
## yet implemented (Logistic Regression, Decision Tree, Random Forest,
## AdaBoost, Gradient Boosting, XGBoost — all pending, later stage).

## Pipeline

```
Raw Data (50,000 rows x 32 columns)
 ↓
Cleaning            (dedupe check, category/range validation — see src/pipeline.py::clean_data)
 ↓
EDA                 (univariate, bivariate, correlation, outliers — see outputs/figures/)
 ↓
Feature Engineering (SGPA_Trend, Practical_Experience_Score, Has_Internship, Avg_Test_Score)
 ↓
Encoding            (OneHotEncoder for nominal, OrdinalEncoder for CollegeTier/CGPA_Tier)
 ↓
Missing Value Handling (median for numeric, most-frequent for categorical)
 ↓
Scaling             (StandardScaler, fit on TRAIN only)
 ↓
PCA                 (25 components, 95.4% variance retained, fit on TRAIN only)
```

Encoding, imputation, and scaling are combined into a single reusable
`sklearn` `ColumnTransformer`, fit on the training split only, then chained
with PCA into one `Pipeline` object saved to
`artifacts/preprocessing_pca_pipeline.joblib`. This is the exact object the
next stage (model training) should load and reuse — do not refit it on the
full dataset.

## Dataset

`data/placement_predict_50k_Dataset1.csv` — 50,000 students, 32 columns.
Target: `PlacementStatus` (binary). `Salary Package` and `IsAnomaly` are
excluded from the feature set (see `src/pipeline.py` for rationale);
`StudentID` is an identifier and also excluded.

## How to run

```bash
pip install -r requirements.txt
python src/pipeline.py
```

This prints the full data understanding / EDA / PCA report to stdout and
writes:

- `outputs/figures/*.png` — 9 EDA + PCA charts
- `outputs/train_pca.csv`, `outputs/test_pca.csv` — PCA-transformed features + target
- `artifacts/preprocessing_pca_pipeline.joblib` — fitted ColumnTransformer + PCA pipeline
- `artifacts/feature_metadata.joblib` — feature name lists used to build the pipeline

## Legacy code

`src/legacy_complete_code.py` is the original exploratory script from before
this pass. It is kept for reference only and is not part of the current
pipeline (it had a broken import, an undefined `load` call, and fit its
scalers on the full dataset rather than train-only — all fixed in
`src/pipeline.py`).

## Web app

No Streamlit or Flask app exists yet in this project (the previously
included `app.py` was actually the Flask library's own source file, not
application code, and has been removed). Building a real app is out of
scope until a trained classifier exists to serve.

## Next stage (not yet started)

Train and compare classifiers (Logistic Regression, Decision Tree, Random
Forest, AdaBoost, Gradient Boosting, XGBoost) on `outputs/train_pca.csv` /
`outputs/test_pca.csv`, or by loading `artifacts/preprocessing_pca_pipeline.joblib`
directly on raw feature columns.
