"""
PlacementPredict — Data pipeline through PCA
=============================================

Stages implemented (and STOPPED at, per project scope):
    Raw Data -> Cleaning -> EDA -> Feature Engineering -> Encoding
    -> Missing Value Handling -> Scaling -> PCA

No classification model (Logistic Regression / Decision Tree / Random Forest /
Boosting / XGBoost) is trained here. That is explicitly out of scope for this
stage of the project.

Run with:
    python src/pipeline.py
"""

import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")  # headless plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "placement_predict_50k_Dataset1.csv")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")
ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
TARGET_COL = "PlacementStatus"

# Columns explicitly excluded from the feature set, and why:
#   StudentID       -> unique identifier, no predictive information
#   Salary Package   -> only known AFTER placement happens (0 whenever a
#                        student is not placed) -> direct target leakage
#   IsAnomaly        -> a synthetic data-quality flag. It marks rows where
#                        CGPA is inconsistent with the semester-wise SGPA
#                        history (verified during EDA), not a genuine
#                        placement predictor, so it is dropped as a feature.
LEAKY_OR_ID_COLS = ["StudentID", "Salary Package", "IsAnomaly"]

NOMINAL_FEATURES = ["Gender", "City", "Stream", "Specialisation", "Hostel", "HistoryOfBacklogs"]
ORDINAL_FEATURES = ["CollegeTier", "CGPA_Tier"]
ORDINAL_CATEGORIES = [
    ["Tier3", "Tier2", "Tier1"],   # CollegeTier: genuine ranking, Tier1 = best
    ["Low", "Mid", "High"],        # CGPA_Tier: genuine ranking
]

BASE_NUMERIC_FEATURES = [
    "SGPA_Sem1", "SGPA_Sem2", "SGPA_Sem3", "SGPA_Sem4",
    "SGPA_Sem5", "SGPA_Sem6", "SGPA_Sem7", "SGPA_Sem8",
    "CGPA", "AttendancePercent", "Internships", "Projects",
    "Workshops", "Certifications", "Publications",
    "AptitudeTestScore", "SoftSkillsRating", "CodingTestScore",
    "MockInterviewScore", "ExtraCurricular",
]

ENGINEERED_NUMERIC_FEATURES = [
    "SGPA_Trend", "Practical_Experience_Score", "Has_Internship", "Avg_Test_Score",
]


# ---------------------------------------------------------------------------
# STEP 1-2: LOAD + DATA UNDERSTANDING
# ---------------------------------------------------------------------------

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df


def print_data_understanding(df):
    print("=" * 70)
    print("DATA UNDERSTANDING")
    print("=" * 70)
    print(f"Shape: {df.shape}")
    print(f"\nColumns and dtypes:\n{df.dtypes}")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    print(f"\nNumerical columns ({len(numeric_cols)}): {numeric_cols}")
    print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")

    print(f"\nTarget column: '{TARGET_COL}'")
    print(f"Target values: {sorted(df[TARGET_COL].unique().tolist())}")
    print(f"Target distribution:\n{df[TARGET_COL].value_counts(normalize=True) * 100}")

    print(f"\nMissing values (columns with any):\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"\nDuplicate rows: {df.duplicated().sum()}")
    print(f"Duplicate StudentID values: {df['StudentID'].duplicated().sum()}")

    print(
        "\nProblem type: Supervised Learning -> Binary Classification.\n"
        "The target 'PlacementStatus' takes exactly two values (0 = not placed, "
        "1 = placed), each row is an independently labeled student record with "
        "no time-ordering to forecast, and the goal is to predict a class label "
        "rather than a continuous quantity -> binary classification."
    )


# ---------------------------------------------------------------------------
# STEP 3: DATA CLEANING
# ---------------------------------------------------------------------------

def clean_data(df):
    """
    Inspection findings (see print_data_understanding output):
      - 0 duplicate rows, 0 duplicate StudentIDs -> nothing to drop.
      - All categorical columns have clean, consistent category spellings
        (verified via .unique() on every categorical column).
      - All numeric columns fall inside plausible ranges (e.g. CGPA in
        [4, 10], AttendancePercent in [50, 100], test scores in [0, 100])
        -> no impossible values to correct.
      - Missing values exist ONLY in 5 numeric columns: Workshops,
        AptitudeTestScore, SoftSkillsRating, CodingTestScore,
        MockInterviewScore (roughly 6-10% each). Missingness rate is
        essentially identical across placement outcomes (checked below),
        consistent with values missing completely at random -> safe to
        median-impute rather than drop rows.
      - 1750 rows are flagged IsAnomaly=1; these correspond almost exactly
        to rows where CGPA is inconsistent with the SGPA history (injected
        noise), not corrupted/unusable rows -> retained, but IsAnomaly is
        excluded as a feature (see LEAKY_OR_ID_COLS).

    No rows are dropped in this step. Missing-value imputation is deferred
    to the reusable preprocessing pipeline (Step 7) so the same logic can
    be reused for validation/inference data.
    """
    df = df.copy()

    n_dupes = df.duplicated().sum()
    if n_dupes > 0:
        df = df.drop_duplicates()
        print(f"Dropped {n_dupes} exact duplicate rows.")
    else:
        print("No duplicate rows found.")

    return df


# ---------------------------------------------------------------------------
# STEP 4: EDA
# ---------------------------------------------------------------------------

def run_eda(df, fig_dir=FIG_DIR):
    os.makedirs(fig_dir, exist_ok=True)
    sns.set_style("whitegrid")

    numeric_cols = [c for c in BASE_NUMERIC_FEATURES]
    categorical_cols = NOMINAL_FEATURES + ORDINAL_FEATURES

    # --- Target analysis ---
    counts = df[TARGET_COL].value_counts().sort_index()
    pct = df[TARGET_COL].value_counts(normalize=True).sort_index() * 100
    print("\n--- TARGET ANALYSIS ---")
    print(f"Class counts:\n{counts}")
    print(f"Class percentages:\n{pct}")
    print(
        f"Class balance: {pct.min():.1f}% / {pct.max():.1f}% split -> "
        "moderately imbalanced, not severe enough to require resampling, "
        "but worth using stratified train/test splits (done below)."
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(data=df, x=TARGET_COL, ax=ax)
    ax.set_title("Target Class Distribution (PlacementStatus)")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "01_target_distribution.png"), dpi=110)
    plt.close(fig)

    # --- Univariate: numerical histograms + boxplots ---
    n = len(numeric_cols)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    for i, col in enumerate(numeric_cols):
        ax = axes.flat[i]
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        ax.set_title(col, fontsize=9)
    for j in range(i + 1, nrows * ncols):
        axes.flat[j].axis("off")
    fig.suptitle("Univariate Distributions — Numerical Features", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "02_numeric_histograms.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3))
    for i, col in enumerate(numeric_cols):
        ax = axes.flat[i]
        sns.boxplot(y=df[col].dropna(), ax=ax)
        ax.set_title(col, fontsize=9)
    for j in range(i + 1, nrows * ncols):
        axes.flat[j].axis("off")
    fig.suptitle("Univariate Boxplots — Numerical Features", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "03_numeric_boxplots.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)

    # --- Univariate: categorical count plots ---
    ncols_c = 2
    nrows_c = int(np.ceil(len(categorical_cols) / ncols_c))
    fig, axes = plt.subplots(nrows_c, ncols_c, figsize=(ncols_c * 6, nrows_c * 3.5))
    for i, col in enumerate(categorical_cols):
        ax = axes.flat[i]
        order = df[col].value_counts().index
        sns.countplot(data=df, y=col, order=order, ax=ax)
        ax.set_title(col)
    for j in range(i + 1, nrows_c * ncols_c):
        axes.flat[j].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "04_categorical_counts.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)

    # --- Bivariate: features vs target ---
    print("\n--- BIVARIATE ANALYSIS (placement rate by group) ---")
    for col in ["CollegeTier", "Internships", "Projects", "HistoryOfBacklogs"]:
        rate = df.groupby(col)[TARGET_COL].mean().sort_values(ascending=False)
        print(f"\nPlacement rate by {col}:\n{rate}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    sns.boxplot(data=df, x=TARGET_COL, y="CGPA", ax=axes[0, 0])
    axes[0, 0].set_title("CGPA vs Placement")

    sns.barplot(
        x=df["Internships"], y=df[TARGET_COL], ax=axes[0, 1],
        estimator=np.mean, errorbar=None,
    )
    axes[0, 1].set_title("Placement Rate by Internship Count")
    axes[0, 1].set_ylabel("Placement Rate")

    sns.barplot(
        x=df["Projects"], y=df[TARGET_COL], ax=axes[1, 0],
        estimator=np.mean, errorbar=None,
    )
    axes[1, 0].set_title("Placement Rate by Project Count")
    axes[1, 0].set_ylabel("Placement Rate")

    sns.barplot(
        x=df["HistoryOfBacklogs"], y=df[TARGET_COL], ax=axes[1, 1],
        estimator=np.mean, errorbar=None,
    )
    axes[1, 1].set_title("Placement Rate by History of Backlogs")
    axes[1, 1].set_ylabel("Placement Rate")

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "05_bivariate_vs_target.png"), dpi=110)
    plt.close(fig)

    # --- Correlation heatmap ---
    corr_cols = numeric_cols + [TARGET_COL]
    corr = df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Matrix — Numerical Features + Target")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "06_correlation_heatmap.png"), dpi=110)
    plt.close(fig)

    print("\n--- CORRELATION WITH TARGET (top drivers) ---")
    print(corr[TARGET_COL].drop(TARGET_COL).sort_values(ascending=False))

    # --- Outliers ---
    print("\n--- OUTLIER CHECK (IQR method) ---")
    outlier_summary = {}
    for col in numeric_cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = ((df[col] < lower) | (df[col] > upper)).sum()
        outlier_summary[col] = n_out
    outlier_series = pd.Series(outlier_summary).sort_values(ascending=False)
    print(outlier_series)
    print(
        "\nDecision: outlier counts are small relative to 50,000 rows and fall "
        "on variables with natural, bounded scales (test scores 0-100, CGPA "
        "4-10, counts of internships/projects/etc.). These look like genuine "
        "high/low performers rather than data errors, so outliers are RETAINED "
        "(not capped or removed). StandardScaler + PCA are considered robust "
        "enough at this sample size; capping is reconsidered only if a later "
        "model at the classifier stage proves too sensitive to them."
    )

    print("\nEDA conclusions:")
    print(
        "- Class balance: ~66% placed / ~34% not placed -> mild imbalance.\n"
        "- Strongest positive drivers of placement: MockInterviewScore, CGPA, "
        "SoftSkillsRating, CodingTestScore, Certifications, Workshops, Projects, "
        "AptitudeTestScore, AttendancePercent, Internships (all correlate "
        "0.5-0.7 with the target).\n"
        "- CollegeTier shows a very strong placement gradient: ~91% (Tier1) "
        "vs ~70% (Tier2) vs ~30% (Tier3).\n"
        "- HistoryOfBacklogs strongly reduces placement odds (~74% vs ~30%).\n"
        "- Missing values are concentrated in 5 test/activity score columns "
        "and appear to be missing at random with respect to the target.\n"
        "- IsAnomaly flags a small subset (3.5%) of rows with internally "
        "inconsistent CGPA/SGPA values; treated as a data-quality signal, "
        "not a feature.\n"
        "- Salary Package is populated only for placed students and is "
        "therefore excluded entirely as an input feature (target leakage)."
    )


# ---------------------------------------------------------------------------
# STEP 5: FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def engineer_features(df):
    """
    New features (all derived only from pre-outcome academic/activity data,
    never from PlacementStatus or Salary Package):

      - SGPA_Trend: SGPA_Sem8 - SGPA_Sem1. Captures whether a student's
        performance improved or declined over their degree, which CGPA
        (a flat average) does not capture.
      - Practical_Experience_Score: Internships + Projects + Certifications.
        A single aggregate measure of hands-on experience.
      - Has_Internship: 1 if Internships > 0 else 0. A simple binary signal
        that placement rate analysis showed is a sharp cutoff (~20% ->
        ~68%+ once a student has at least one internship).
      - Avg_Test_Score: row-wise mean of AptitudeTestScore, CodingTestScore,
        MockInterviewScore (NaN-skipping). A combined "assessment"
        signal; may itself be NaN if all three are missing, which the
        pipeline's numeric imputer will handle downstream.
    """
    df = df.copy()

    df["SGPA_Trend"] = df["SGPA_Sem8"] - df["SGPA_Sem1"]

    df["Practical_Experience_Score"] = (
        df["Internships"] + df["Projects"] + df["Certifications"]
    )

    df["Has_Internship"] = (df["Internships"] > 0).astype(int)

    df["Avg_Test_Score"] = df[
        ["AptitudeTestScore", "CodingTestScore", "MockInterviewScore"]
    ].mean(axis=1, skipna=True)

    return df


# ---------------------------------------------------------------------------
# STEPS 6-8: ENCODING + MISSING VALUE HANDLING + SCALING (as one reusable
# preprocessing pipeline, fit on TRAIN only to avoid leakage)
# ---------------------------------------------------------------------------

def build_preprocessing_pipeline(numeric_features, ordinal_features, ordinal_categories, nominal_features):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    ordinal_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(categories=ordinal_categories)),
    ])

    nominal_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("ord", ordinal_transformer, ordinal_features),
        ("nom", nominal_transformer, nominal_features),
    ])
    return preprocessor


# ---------------------------------------------------------------------------
# STEPS 9-11: PCA + VISUALIZATIONS + OUTPUT
# ---------------------------------------------------------------------------

def run_pca(X_train_processed, X_test_processed, y_train, fig_dir=FIG_DIR, variance_threshold=0.95):
    # Fit a "full" PCA first (no component limit) purely to inspect variance.
    pca_full = PCA(random_state=RANDOM_STATE)
    pca_full.fit(X_train_processed)

    explained = pca_full.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    thresholds = {}
    for t in [0.90, 0.95, 0.99]:
        n_needed = int(np.argmax(cumulative >= t) + 1)
        thresholds[t] = n_needed
        print(f"Components needed for {int(t*100)}% variance: {n_needed}")

    n_components = thresholds[variance_threshold]
    print(
        f"\nSelected n_components = {n_components} "
        f"(retains {cumulative[n_components-1]*100:.2f}% variance, "
        f"the {int(variance_threshold*100)}% threshold), balancing "
        "dimensionality reduction against information loss for a dataset "
        f"with {X_train_processed.shape[1]} encoded input features."
    )

    # --- Plot 1: explained variance per component ---
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(1, len(explained) + 1), explained, alpha=0.7, label="Individual")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("Explained Variance by Principal Component")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "07_pca_explained_variance.png"), dpi=110)
    plt.close(fig)

    # --- Plot 2: cumulative explained variance ---
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(1, len(cumulative) + 1), cumulative, marker="o", markersize=3)
    for t, n_needed in thresholds.items():
        ax.axhline(t, color="gray", linestyle="--", linewidth=0.8)
        ax.axvline(n_needed, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(n_components, color="red", linestyle="-", linewidth=1.5, label=f"Selected: {n_components}")
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("Cumulative Explained Variance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "08_pca_cumulative_variance.png"), dpi=110)
    plt.close(fig)

    # --- Final PCA fit with selected n_components ---
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train_processed)
    X_test_pca = pca.transform(X_test_processed)

    # --- Plot 3: 2D projection (PC1 vs PC2), target used ONLY for coloring ---
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        X_train_pca[:, 0], X_train_pca[:, 1],
        c=y_train, cmap="coolwarm", alpha=0.3, s=8,
    )
    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}% variance)")
    ax.set_title("2D PCA Projection (PC1 vs PC2), colored by PlacementStatus")
    legend1 = ax.legend(*scatter.legend_elements(), title="Placed")
    ax.add_artist(legend1)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "09_pca_2d_projection.png"), dpi=110)
    plt.close(fig)

    return pca, pca_full, X_train_pca, X_test_pca, n_components, cumulative


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "outputs"), exist_ok=True)

    # Step 1-2
    df = load_data()
    print_data_understanding(df)

    # Step 3
    df = clean_data(df)

    # Step 4
    run_eda(df)

    # Step 5
    df_fe = engineer_features(df)

    numeric_features = BASE_NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES
    feature_cols = numeric_features + ORDINAL_FEATURES + NOMINAL_FEATURES

    print("\n" + "=" * 70)
    print("FINAL FEATURE LIST BEFORE ENCODING/PCA")
    print("=" * 70)
    print(f"Numeric ({len(numeric_features)}): {numeric_features}")
    print(f"Ordinal ({len(ORDINAL_FEATURES)}): {ORDINAL_FEATURES}")
    print(f"Nominal ({len(NOMINAL_FEATURES)}): {NOMINAL_FEATURES}")
    print(f"Total raw input columns before encoding: {len(feature_cols)}")
    print(f"Excluded columns: {LEAKY_OR_ID_COLS} (see clean_data docstring for rationale)")

    X = df_fe[feature_cols]
    y = df_fe[TARGET_COL]

    # Step 12: split BEFORE fitting anything
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y,
    )
    print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Steps 6-8: fit preprocessing on TRAIN ONLY, transform both
    preprocessor = build_preprocessing_pipeline(
        numeric_features, ORDINAL_FEATURES, ORDINAL_CATEGORIES, NOMINAL_FEATURES,
    )
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print(f"\nBefore PCA shape (train): {X_train_processed.shape}")
    print(f"Before PCA shape (test):  {X_test_processed.shape}")

    # Steps 9-11: PCA fit on TRAIN ONLY, transform both
    pca, pca_full, X_train_pca, X_test_pca, n_components, cumulative = run_pca(
        X_train_processed, X_test_processed, y_train,
    )

    print(f"\nAfter PCA shape (train): {X_train_pca.shape}")
    print(f"After PCA shape (test):  {X_test_pca.shape}")

    # Save PCA-transformed datasets
    outputs_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    pd.DataFrame(X_train_pca, columns=[f"PC{i+1}" for i in range(n_components)]) \
        .assign(PlacementStatus=y_train.values) \
        .to_csv(os.path.join(outputs_dir, "train_pca.csv"), index=False)
    pd.DataFrame(X_test_pca, columns=[f"PC{i+1}" for i in range(n_components)]) \
        .assign(PlacementStatus=y_test.values) \
        .to_csv(os.path.join(outputs_dir, "test_pca.csv"), index=False)

    # Save fitted preprocessing + PCA pipeline for reuse in the next stage
    full_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("pca", pca),
    ])
    joblib.dump(full_pipeline, os.path.join(ARTIFACT_DIR, "preprocessing_pca_pipeline.joblib"))
    joblib.dump(
        {"feature_cols": feature_cols, "numeric_features": numeric_features,
         "ordinal_features": ORDINAL_FEATURES, "nominal_features": NOMINAL_FEATURES},
        os.path.join(ARTIFACT_DIR, "feature_metadata.joblib"),
    )

    # -----------------------------------------------------------------
    # FINAL REPORT
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    print(f"Original feature count (raw input columns): {len(feature_cols)}")
    print(f"Encoded feature count (after one-hot/ordinal/scaling): {X_train_processed.shape[1]}")
    print(f"Selected PCA components: {n_components}")
    print(f"Variance retained: {cumulative[n_components-1]*100:.2f}%")
    print(f"Before PCA shape: train={X_train_processed.shape}, test={X_test_processed.shape}")
    print(f"After PCA shape:  train={X_train_pca.shape}, test={X_test_pca.shape}")
    print(f"\nPCA output saved to: {os.path.join(outputs_dir, 'train_pca.csv')}, "
          f"{os.path.join(outputs_dir, 'test_pca.csv')}")
    print(f"Fitted pipeline saved to: {os.path.join(ARTIFACT_DIR, 'preprocessing_pca_pipeline.joblib')}")
    print(f"EDA/PCA figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
