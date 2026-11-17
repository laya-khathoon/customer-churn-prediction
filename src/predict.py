"""Predict churn for one customer with the saved model.

Run as a script from the repo root:

    python src/predict.py

The loading and prediction steps are functions so they can be
imported and tested offline; running the file directly behaves
exactly as before.
"""

import joblib
import pandas as pd


# ==========================================
# 1. Load saved model
# ==========================================

def load_model(path="models/churn_model.pkl"):
    """Load the pickle with preprocessor, model and threshold."""
    return joblib.load(path)


# ==========================================
# 2. Predict churn probability for one customer
# ==========================================

def predict_customer(customer, model_data):
    """Return (churn_probability, prediction) for a one-row DataFrame."""
    customer_processed = model_data["preprocessor"].transform(
        customer
    )

    churn_probability = model_data["model"].predict_proba(
        customer_processed
    )[:, 1][0]

    prediction = (
        churn_probability >= model_data["threshold"]
    )

    return churn_probability, bool(prediction)


# ==========================================
# Sample customer used by the script
# ==========================================

SAMPLE_CUSTOMER = pd.DataFrame([
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
# Script entry point (same output as before)
# ==========================================

def main():
    model_data = load_model()
    threshold = model_data["threshold"]

    churn_probability, prediction = predict_customer(
        SAMPLE_CUSTOMER,
        model_data
    )

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


if __name__ == "__main__":
    main()
