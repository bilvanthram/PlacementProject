import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
    roc_curve,
    roc_auc_score
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix


# =========================================================
# 1. LOAD DATA
# =========================================================

def load_data():
    # Loads the CSV from the same folder as this script,
    # so it works no matter where you run it from.
    here = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(here, "placement_predict_50k_Dataset1.csv")
    df = pd.read_csv(csv_path)
    return df


# =========================================================
# 2. SUMMARY
# =========================================================

def summarize(df):
    print("\n========== DATA SUMMARY ==========")
    print("\n--- Describe ---")
    print(df.describe())
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    print("\n--- Shape ---")
    print(df.shape)
    print("\n--- First 5 Rows ---")
    print(df.head())


# =========================================================
# 3. STATISTICAL ANALYSIS
# =========================================================

def stat(df):
    print("\n========== STATISTICS ==========")

    num_cols = df.select_dtypes(include=np.number)

    print("\nNumerical Columns:")
    print(num_cols.columns.tolist())

    mean_values = num_cols.mean()
    print("\nMean:")
    print(mean_values)

    std_values = num_cols.std()
    print("\nStandard Deviation:")
    print(std_values)

    skewness = num_cols.skew()
    print("\nSkewness:")
    print(skewness)

    kurtosis = num_cols.kurt()
    print("\nKurtosis:")
    print(kurtosis)

    correlation = num_cols.corr()
    print("\nCorrelation:")
    print(correlation)

    return num_cols, mean_values, std_values, skewness, kurtosis, correlation


# =========================================================
# 4. VISUALIZATION
# =========================================================

def visualize(df):

    numeric_df = df.select_dtypes(include=np.number)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="YlGnBu",
        ax=axes[0, 0]
    )
    axes[0, 0].set_title("Correlation Heatmap")

    if "PlacementStatus" in df.columns:
        sns.countplot(
            data=df,
            x="PlacementStatus",
            ax=axes[0, 1]
        )
        axes[0, 1].set_title("Placement Status")

    if "Salary Package" in df.columns:
        sns.boxplot(
            y=df["Salary Package"],
            ax=axes[0, 2]
        )
        axes[0, 2].set_title("Salary Package")

    if "CGPA" in df.columns:
        sns.histplot(
            df["CGPA"],
            kde=True,
            ax=axes[1, 0]
        )
    axes[1, 0].set_title("CGPA Distribution")

    if "CGPA" in df.columns:
        sns.boxplot(
            y=df["CGPA"],
            ax=axes[1, 1]
        )
    axes[1, 1].set_title("CGPA Boxplot")

    if "CGPA" in df.columns and "Salary Package" in df.columns:
        sns.scatterplot(
            data=df,
            x="CGPA",
            y="Salary Package",
            ax=axes[1, 2]
        )
    axes[1, 2].set_title("CGPA vs Salary Package")

    plt.tight_layout()
    plt.show()


# =========================================================
# 5. NORMALIZATION AND STANDARDIZATION
# =========================================================

def normalize(df, exclude_cols=None):

    print("\n========== SCALING ==========")

    if exclude_cols is None:
        exclude_cols = []

    numeric_cols_all = df.select_dtypes(include=np.number).columns
    scale_cols = [c for c in numeric_cols_all if c not in exclude_cols]

    numeric_df = df[scale_cols].copy()
    keep_df = df.drop(columns=scale_cols).copy()

    print("\nOriginal Data:")
    print(df.head())

    # Min-Max Normalization
    min_max_scaler = MinMaxScaler()

    normalized_array = min_max_scaler.fit_transform(numeric_df)

    normalized_numeric_df = pd.DataFrame(
        normalized_array,
        columns=numeric_df.columns,
        index=numeric_df.index
    )

    normalized_df = pd.concat(
        [keep_df, normalized_numeric_df],
        axis=1
    )[df.columns]

    print("\n========== MIN-MAX NORMALIZED DATA ==========")
    print(normalized_df.head())

    # Standardization
    standard_scaler = StandardScaler()

    standardized_array = standard_scaler.fit_transform(numeric_df)

    standardized_numeric_df = pd.DataFrame(
        standardized_array,
        columns=numeric_df.columns,
        index=numeric_df.index
    )

    standardized_df = pd.concat(
        [keep_df, standardized_numeric_df],
        axis=1
    )[df.columns]

    print("\n========== STANDARDIZED DATA ==========")
    print(standardized_df.head())

    return normalized_df, standardized_df


# =========================================================
# 6. OUTLIER HANDLING
# =========================================================

def outlier(df):

    if "Salary Package" not in df.columns:
        print("Salary Package column not found.")
        return df

    df = df.copy()

    print("\n========== OUTLIER HANDLING ==========")

    print("\nBefore Clipping:")
    print(df["Salary Package"].describe())

    q1 = df["Salary Package"].quantile(0.25)
    q3 = df["Salary Package"].quantile(0.75)

    iqr = q3 - q1

    upper = q3 + 1.5 * iqr

    df["Salary Package"] = df["Salary Package"].clip(upper=upper)

    print("\nAfter Clipping:")
    print(df["Salary Package"].describe())

    return df


# =========================================================
# 7. ONE-HOT ENCODING
# =========================================================

def onehot(df):

    nominal_cols = [
        "Gender",
        "City",
        "Stream",
        "Specialisation",
        "Hostel",
        "HistoryOfBacklogs"
    ]

    nominal_cols = [col for col in nominal_cols if col in df.columns]

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["PlacementStatus"]
    )

    ohe = OneHotEncoder(
        drop="first",
        sparse_output=False,
        handle_unknown="ignore"
    )

    train_ohe = ohe.fit_transform(train_df[nominal_cols])
    test_ohe = ohe.transform(test_df[nominal_cols])

    ohe_cols = ohe.get_feature_names_out(nominal_cols)

    train_ohe_df = pd.DataFrame(
        train_ohe,
        columns=ohe_cols,
        index=train_df.index
    )

    test_ohe_df = pd.DataFrame(
        test_ohe,
        columns=ohe_cols,
        index=test_df.index
    )

    print("\nTraining Data:")
    print(train_ohe_df.head())

    print("\nTesting Data:")
    print(test_ohe_df.head())

    return train_ohe_df, test_ohe_df


# =========================================================
# 8. GRADIENT DESCENT
# =========================================================

def gradient_descent(X, y, learning_rate=0.001, iterations=1000):

    X = np.asarray(X, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    m = len(y)

    theta0 = 0.0
    theta1 = 0.0

    mse_history = []

    for _ in range(iterations):

        y_pred = theta0 + theta1 * X

        error = y_pred - y

        mse = np.mean(error ** 2)
        mse_history.append(mse)

        d_theta0 = (2 / m) * np.sum(error)
        d_theta1 = (2 / m) * np.sum(error * X)

        theta0 -= learning_rate * d_theta0
        theta1 -= learning_rate * d_theta1

    return theta0, theta1, mse_history


# =========================================================
# 9. SIMPLE LINEAR REGRESSION USING GRADIENT DESCENT
# =========================================================

def simple_linear_regression_GD(df):

    print("\n========== LINEAR REGRESSION USING GRADIENT DESCENT ==========")

    X = df[["CGPA"]]
    y = df["Salary Package"]

    train_X, test_X, train_y, test_y = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    theta0, theta1, mse_history = gradient_descent(
        train_X,
        train_y,
        learning_rate=0.001,
        iterations=1000
    )

    print("Theta0 (Intercept):", theta0)
    print("Theta1 (Slope):", theta1)

    test_X_array = test_X["CGPA"].values
    y_pred = theta0 + theta1 * test_X_array

    mse = mean_squared_error(test_y, y_pred)
    r2 = r2_score(test_y, y_pred)

    print("MSE:", mse)
    print("R-Square Score:", r2)

    plt.figure(figsize=(8, 5))
    plt.plot(mse_history)
    plt.xlabel("Iterations")
    plt.ylabel("MSE")
    plt.title("Gradient Descent: MSE vs Iterations")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.scatter(
        test_X["CGPA"],
        test_y,
        alpha=0.5,
        label="Actual"
    )

    order = np.argsort(test_X["CGPA"].values)

    plt.plot(
        test_X["CGPA"].values[order],
        y_pred[order],
        linewidth=2,
        label="Regression Line"
    )

    plt.xlabel("CGPA")
    plt.ylabel("Salary Package")
    plt.title("CGPA vs Salary Package")
    plt.legend()
    plt.grid(True)
    plt.show()


# =========================================================
# 10. SIMPLE LINEAR REGRESSION USING SKLEARN
# =========================================================

def simple_linear_regression(df):

    from sklearn.linear_model import LinearRegression

    print("\n========== SKLEARN LINEAR REGRESSION ==========")

    X = df[["CGPA"]]
    y = df["Salary Package"]

    train_X, test_X, train_y, test_y = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()

    model.fit(train_X, train_y)

    y_pred = model.predict(test_X)

    mse = mean_squared_error(test_y, y_pred)
    r2 = r2_score(test_y, y_pred)

    print("Intercept:", model.intercept_)
    print("Coefficient:", model.coef_[0])
    print("MSE:", mse)
    print("R-Square Score:", r2)

    return model


# =========================================================
# 11. DECISION TREE + ROC + AUC
# =========================================================

def decision_tree_pruned(df):

    x = df[["CGPA", "Internships", "AptitudeTestScore"]]
    y = df["PlacementStatus"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )

    # Train
    model.fit(x_train, y_train)

    # Prediction
    y_pred = model.predict(x_test)

    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Precision and Recall
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0

    # F1 Score
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) != 0 else 0)

    # F2 Score
    f2 = (5 * precision * recall / (4 * precision + recall)
          if (4 * precision + recall) != 0 else 0
          )

    # Probability for ROC/AUC
    y_prob = model.predict_proba(x_test)[:, 1]

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # AUC
    auc = roc_auc_score(y_test, y_prob)

    print("\n========== PRUNED DECISION TREE ==========")
    print("Accuracy:", accuracy)
    print("AUC:", auc)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\n========== CONFUSION MATRIX ==========")
    print("True Positive  (TP):", tp)
    print("True Negative  (TN):", tn)
    print("False Positive (FP):", fp)
    print("False Negative (FN):", fn)

    print("\n========== PERFORMANCE METRICS ==========")
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("F2 Score:", f2)
    print("ROC-AUC:", auc)



    # =====================================================
    # ROC CURVE
    # =====================================================

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    plt.figure(figsize=(8, 6))

    plt.plot(
        fpr,
        tpr,
        label=f"Decision Tree (AUC = {auc:.3f})"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Decision Tree")
    plt.legend()
    plt.grid(True)

    plt.show()

    return model
# =========================================================
# MAIN PROGRAM
# =========================================================

if __name__ == "__main__":

    df = load_data()

    print("Dataset Shape:", df.shape)
    print(df.head())

    # summarize(df)
    # stat(df)
    # visualize(df)

    target_cols = ["PlacementStatus", "IsAnomaly", "Salary Package"]

    normalized_df, standardized_df = normalize(df, target_cols)

    standardized_df = outlier(standardized_df)

    # train_ohe_df, test_ohe_df = onehot(standardized_df)

    simple_linear_regression_GD(df)

    # simple_linear_regression(df)

    decision_tree_pruned(df)