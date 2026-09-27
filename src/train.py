import pandas as pd
import joblib
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


# ==========================================
# 1. Load dataset
# ==========================================

df = pd.read_csv("data/customer_churn.csv")


# ==========================================
# 2. Convert TotalCharges to numeric
# ==========================================

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)


# ==========================================
# 3. Remove customer ID
# ==========================================

df = df.drop("customerID", axis=1)


# ==========================================
# 4. Separate features and target
# ==========================================

X = df.drop("Churn", axis=1)

y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})


# ==========================================
# 5. Split into training, validation and testing
# ==========================================

# 80% training + validation
# 20% testing

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# 64% training + 16% validation
# from the original dataset

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.20,
    random_state=42,
    stratify=y_train_val
)


# ==========================================
# 6. Identify feature types
# ==========================================

numerical_features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

categorical_features = [
    column
    for column in X.columns
    if column not in numerical_features
]


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


# ==========================================
# 7. Numerical preprocessing
# ==========================================

numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


# ==========================================
# 8. Categorical preprocessing
# ==========================================

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)


# ==========================================
# 9. Combine preprocessing pipelines
# ==========================================

preprocessor = ColumnTransformer(
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
# 10. Process training, validation and testing
# ==========================================

X_train_processed = preprocessor.fit_transform(
    X_train
)

X_val_processed = preprocessor.transform(
    X_val
)

X_test_processed = preprocessor.transform(
    X_test
)


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


# ==========================================
# 11. Train XGBoost model
# ==========================================

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


print("\nXGBoost model training completed.")


# ==========================================
# 12. Validation predictions
# ==========================================

y_val_probability = model.predict_proba(
    X_val_processed
)[:, 1]


print("\nFirst 10 validation probabilities:")
print(y_val_probability[:10])


# ==========================================
# 13. Select best threshold using validation set
# ==========================================

print("\nValidation Threshold Analysis")
print("-------------------------")

thresholds = [
    0.3,
    0.4,
    0.5,
    0.6,
    0.7
]

best_threshold = 0.5
best_f1 = 0


for threshold in thresholds:

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

    if threshold_f1 > best_f1:
        best_f1 = threshold_f1
        best_threshold = threshold


print(
    f"\nSelected threshold: {best_threshold:.1f}"
)

print(
    f"Validation F1 Score: {best_f1:.4f}"
)


# ==========================================
# 14. Final test predictions
# ==========================================

y_probability = model.predict_proba(
    X_test_processed
)[:, 1]


y_pred = (
    y_probability >= best_threshold
).astype(int)


print("\nFirst 10 test predictions:")
print(y_pred[:10])


print("\nFirst 10 test churn probabilities:")
print(y_probability[:10])


# ==========================================
# 15. Final model evaluation
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


print("\nFinal Model Evaluation")
print("-------------------------")

print(f"Selected Threshold: {best_threshold:.1f}")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")


# ==========================================
# 16. Classification Report
# ==========================================

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


# ==========================================
# 17. Confusion Matrix
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred
)


print("\nConfusion Matrix")
print("-------------------------")

print(cm)
# ==========================================
# 18. Save model and preprocessing pipeline
# ==========================================

model_data = {
    "preprocessor": preprocessor,
    "model": model,
    "threshold": best_threshold
}

joblib.dump(
    model_data,
    "models/churn_model.pkl"
)

print("\nModel saved successfully.")
print("Saved file: models/churn_model.pkl")