import streamlit as st
import pandas as pd
import joblib


# ==========================================
# 1. Page configuration
# ==========================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# ==========================================
# 2. Load trained model
# ==========================================

model_data = joblib.load(
    "models/churn_model.pkl"
)

preprocessor = model_data["preprocessor"]
model = model_data["model"]
threshold = model_data["threshold"]


# ==========================================
# 3. Title
# ==========================================

st.title("📊 Customer Churn Prediction")

st.write(
    "Enter customer information to predict "
    "the probability of customer churn."
)


# ==========================================
# 4. Customer information
# ==========================================

st.header("Customer Information")


col1, col2, col3 = st.columns(3)


with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=12
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )


with col2:

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )


with col3:

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=840.0
    )


# ==========================================
# 5. Prediction button
# ==========================================

st.divider()

if st.button(
    "Predict Churn",
    type="primary"
):

    customer = pd.DataFrame([
        {
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }
    ])


    # ==========================================
    # 6. Preprocess customer
    # ==========================================

    customer_processed = preprocessor.transform(
        customer
    )


    # ==========================================
    # 7. Predict probability
    # ==========================================

    churn_probability = model.predict_proba(
        customer_processed
    )[:, 1][0]


    # ==========================================
    # 8. Apply threshold
    # ==========================================

    prediction = (
        churn_probability >= threshold
    )


    # ==========================================
    # 9. Display result
    # ==========================================

    st.header("Prediction Result")

    probability_percentage = (
        churn_probability * 100
    )

    st.metric(
        "Churn Probability",
        f"{probability_percentage:.2f}%"
    )

    st.write(
        f"Classification threshold: "
        f"**{threshold:.2f}**"
    )


    if prediction:

        st.error(
            "⚠️ Customer is likely to churn."
        )

    else:

        st.success(
            "✅ Customer is likely to stay."
        )