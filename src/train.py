"""Train the customer churn model.

Run as a script from the repo root:

    python src/train.py

Each step is a small function so the pipeline can be imported and
tested offline; running the file directly behaves exactly as before.
"""

import joblib
import pandas as pd
from xgboost import XGBClassifier

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


NUMERICAL_FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]


# ==========================================
# 1. Load dataset
# ==========================================

def load_data(path="data/customer_churn.csv"):
    """Read the CSV, fix TotalCharges and drop the customer ID."""
    df = pd.read_csv(path)

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df = df.drop("customerID", axis=1)
    return df


# ==========================================
# 2. Separate features and target
# ==========================================

def split_features_target(df):
    """Return (X, y) with Churn mapped to 0/1."""
    X = df.drop("Churn", axis=1)

    y = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    return X, y


# ==========================================
# 3. Split into training, validation and testing
# ==========================================

def make_splits(X, y, test_size=0.20, random_state=42):
    """80% train+val / 20% test, then 64% train / 16% val."""
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=test_size,
        random_state=random_state,
        stratify=y_train_val
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def categorical_features_for(X):
    """Every feature that is not one of the numerical ones."""
    return [
        column
        for column in X.columns
        if column not in NUMERICAL_FEATURES
    ]


# ==========================================
# 4. Preprocessing pipelines
# ==========================================

def build_preprocessor(numerical_features, categorical_features):
    """Median-impute + scale numbers, mode-impute + one-hot encode the rest."""
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_pipeline,
                numerical_features
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_features
            )
        ]
    )


# ==========================================
# 5. Train XGBoost model
# ==========================================

def train_model(X_train_processed, y_train):
    """Fit the XGBoost classifier with the project hyperparameters."""
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )

    model.fit(
        X_train_processed,
        y_train
    )

    return model


# ==========================================
# 6. Select best threshold using validation set
# ==========================================

def select_threshold(model, X_val_processed, y_val, thresholds=THRESHOLDS):
    """Pick the threshold with the best validation F1 score."""
    y_val_probability = model.predict_proba(
        X_val_processed
    )[:, 1]

    best_threshold = 0.5
    best_f1 = 0

    for threshold in thresholds:
        y_val_pred = (
            y_val_probability >= threshold
        ).astype(int)

        threshold_f1 = f1_score(
            y_val,
            y_val_pred,
            zero_division=0
        )

        if threshold_f1 > best_f1:
            best_f1 = threshold_f1
            best_threshold = threshold

    return best_threshold, best_f1


# ==========================================
# 7. Final model evaluation
# ==========================================

def evaluate_model(model, X_test_processed, y_test, threshold):
    """Return (metrics, y_pred, y_probability) on the test split."""
    y_probability = model.predict_proba(
        X_test_processed
    )[:, 1]

    y_pred = (
        y_probability >= threshold
    ).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_probability)
    }

    return metrics, y_pred, y_probability


# ==========================================
# 8. Save model and preprocessing pipeline
# ==========================================

def save_model(preprocessor, model, threshold, path="models/churn_model.pkl"):
    """Bundle preprocessor, model and threshold into one pickle."""
    model_data = {
        "preprocessor": preprocessor,
        "model": model,
        "threshold": threshold
    }

    joblib.dump(model_data, path)


# ==========================================
# Script entry point (same output as before)
# ==========================================

def main():
    df = load_data()
    X, y = split_features_target(df)

    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(X, y)

    numerical_features = NUMERICAL_FEATURES
    categorical_features = categorical_features_for(X)

    print("Feature shape:", X.shape)
    print("Target shape:", y.shape)

    print("\nTraining features:", X_train.shape)
    print("Validation features:", X_val.shape)
    print("Testing features:", X_test.shape)

    print("\nNumerical features:")
    print(numerical_features)

    print("\nCategorical features:")
    print(categorical_features)

    print("\nTraining target:")
    print(y_train.value_counts())

    print("\nValidation target:")
    print(y_val.value_counts())

    print("\nTesting target:")
    print(y_test.value_counts())

    preprocessor = build_preprocessor(
        numerical_features,
        categorical_features
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    print(
        "\nProcessed training shape:",
        X_train_processed.shape
    )

    print(
        "Processed validation shape:",
        X_val_processed.shape
    )

    print(
        "Processed testing shape:",
        X_test_processed.shape
    )

    model = train_model(X_train_processed, y_train)

    print("\nXGBoost model training completed.")

    y_val_probability = model.predict_proba(
        X_val_processed
    )[:, 1]

    print("\nFirst 10 validation probabilities:")
    print(y_val_probability[:10])

    print("\nValidation Threshold Analysis")
    print("-------------------------")

    for threshold in THRESHOLDS:
        y_val_pred = (
            y_val_probability >= threshold
        ).astype(int)

        threshold_precision = precision_score(
            y_val,
            y_val_pred,
            zero_division=0
        )

        threshold_recall = recall_score(
            y_val,
            y_val_pred,
            zero_division=0
        )

        threshold_f1 = f1_score(
            y_val,
            y_val_pred,
            zero_division=0
        )

        print(
            f"Threshold: {threshold:.1f} | "
            f"Precision: {threshold_precision:.4f} | "
            f"Recall: {threshold_recall:.4f} | "
            f"F1: {threshold_f1:.4f}"
        )

    best_threshold, best_f1 = select_threshold(
        model,
        X_val_processed,
        y_val
    )

    print(
        f"\nSelected threshold: {best_threshold:.1f}"
    )

    print(
        f"Validation F1 Score: {best_f1:.4f}"
    )

    metrics, y_pred, y_probability = evaluate_model(
        model,
        X_test_processed,
        y_test,
        best_threshold
    )

    print("\nFirst 10 test predictions:")
    print(y_pred[:10])

    print("\nFirst 10 test churn probabilities:")
    print(y_probability[:10])

    print("\nFinal Model Evaluation")
    print("-------------------------")

    print(f"Selected Threshold: {best_threshold:.1f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")

    print("\nClassification Report")
    print("-------------------------")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "No Churn",
                "Churn"
            ]
        )
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print("\nConfusion Matrix")
    print("-------------------------")

    print(cm)

    save_model(preprocessor, model, best_threshold)

    print("\nModel saved successfully.")


if __name__ == "__main__":
    main()
