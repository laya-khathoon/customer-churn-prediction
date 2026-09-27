import pandas as pd
import joblib


# ==========================================
# 1. Load saved model
# ==========================================

model_data = joblib.load(
    "models/churn_model.pkl"
)

preprocessor = model_data["preprocessor"]
model = model_data["model"]
threshold = model_data["threshold"]


# ==========================================
# 2. Create sample customer
# ==========================================

customer = pd.DataFrame([
    {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 80.0,
        "TotalCharges": 960.0
    }
])


# ==========================================
# 3. Preprocess customer data
# ==========================================

customer_processed = preprocessor.transform(
    customer
)


# ==========================================
# 4. Predict churn probability
# ==========================================

churn_probability = model.predict_proba(
    customer_processed
)[:, 1][0]


# ==========================================
# 5. Make prediction
# ==========================================

prediction = (
    churn_probability >= threshold
)


# ==========================================
# 6. Display result
# ==========================================

print("\nCustomer Churn Prediction")
print("-------------------------")

print(
    f"Churn Probability: "
    f"{churn_probability:.2%}"
)

print(
    f"Classification Threshold: "
    f"{threshold:.2f}"
)


if prediction:
    print("Prediction: Customer is likely to churn.")
else:
    print("Prediction: Customer is likely to stay.")