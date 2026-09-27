import streamlit as st
import pandas as pd
import joblib
import textwrap


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ChurnIQ | Customer Churn Prediction",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# LOAD MODEL
# ============================================================

model_data = joblib.load("models/churn_model.pkl")

preprocessor = model_data["preprocessor"]
model = model_data["model"]
threshold = model_data["threshold"]


# ============================================================
# CUSTOM CSS
# ============================================================

st.html(
    textwrap.dedent("""
    <style>

    @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
    );

    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(37, 99, 235, 0.18),
                transparent 28%
            ),
            radial-gradient(
                circle at 90% 20%,
                rgba(6, 182, 212, 0.14),
                transparent 30%
            ),
            radial-gradient(
                circle at 50% 100%,
                rgba(124, 58, 237, 0.12),
                transparent 35%
            ),
            #050914;

        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }


    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 4rem;
        position: relative;
        z-index: 2;
    }


    #MainMenu,
    footer,
    header {
        visibility: hidden;
    }


    /* ======================================================
       ANIMATED BACKGROUND
    ====================================================== */

    .orb {
        position: fixed;
        border-radius: 50%;
        filter: blur(90px);
        opacity: 0.18;
        pointer-events: none;
        z-index: 0;
    }

    .orb-one {
        width: 320px;
        height: 320px;
        background: #2563eb;
        left: -120px;
        top: 5%;
        animation: floatOne 12s ease-in-out infinite;
    }

    .orb-two {
        width: 280px;
        height: 280px;
        background: #06b6d4;
        right: -100px;
        top: 35%;
        animation: floatTwo 15s ease-in-out infinite;
    }

    .orb-three {
        width: 240px;
        height: 240px;
        background: #7c3aed;
        left: 40%;
        bottom: 0;
        animation: floatThree 18s ease-in-out infinite;
    }


    @keyframes floatOne {
        0%, 100% {
            transform: translate(0, 0);
        }

        50% {
            transform: translate(90px, 70px);
        }
    }


    @keyframes floatTwo {
        0%, 100% {
            transform: translate(0, 0);
        }

        50% {
            transform: translate(-90px, 60px);
        }
    }


    @keyframes floatThree {
        0%, 100% {
            transform: translateY(0);
        }

        50% {
            transform: translateY(-90px);
        }
    }


    /* ======================================================
       HERO
    ====================================================== */

    .hero {
        padding: 45px 0 30px;
        animation: heroEnter 0.9s ease-out;
    }


    @keyframes heroEnter {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }

        to {
            opacity: 1;
            transform: translateY(0);
        }
    }


    .eyebrow {
        display: flex;
        align-items: center;
        gap: 9px;
        color: #60a5fa;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 2px;
        margin-bottom: 16px;
    }


    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #22c55e;
        animation: pulse 2s infinite;
    }


    @keyframes pulse {
        0% {
            box-shadow:
                0 0 0 0
                rgba(34, 197, 94, 0.6);
        }

        70% {
            box-shadow:
                0 0 0 10px
                rgba(34, 197, 94, 0);
        }

        100% {
            box-shadow:
                0 0 0 0
                rgba(34, 197, 94, 0);
        }
    }


    .hero-title {
        margin: 0;
        font-size: clamp(42px, 6vw, 72px);
        line-height: 1.03;
        font-weight: 800;
        letter-spacing: -3px;
        color: #f8fafc;
    }


    .gradient-text {
        background:
            linear-gradient(
                90deg,
                #60a5fa,
                #22d3ee,
                #818cf8,
                #60a5fa
            );

        background-size: 300% auto;

        -webkit-background-clip: text;
        background-clip: text;

        -webkit-text-fill-color: transparent;

        animation:
            gradientFlow
            5s
            linear
            infinite;
    }


    @keyframes gradientFlow {
        0% {
            background-position: 0% center;
        }

        100% {
            background-position: 300% center;
        }
    }


    .hero-description {
        max-width: 750px;
        margin-top: 20px;
        color: #94a3b8;
        font-size: 16px;
        line-height: 1.7;
    }


    /* ======================================================
       BADGES
    ====================================================== */

    .badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 25px;
    }


    .badge {
        padding: 9px 15px;
        border-radius: 30px;

        background:
            rgba(15, 23, 42, 0.7);

        border:
            1px solid
            rgba(148, 163, 184, 0.16);

        color: #cbd5e1;
        font-size: 12px;

        backdrop-filter: blur(15px);

        transition:
            all 0.3s ease;
    }


    .badge:hover {
        transform: translateY(-3px);

        border-color:
            rgba(96, 165, 250, 0.5);

        box-shadow:
            0 8px 25px
            rgba(37, 99, 235, 0.15);
    }


    /* ======================================================
       SECTIONS
    ====================================================== */

    .section {
        margin-top: 35px;
    }


    .section-header {
        display: flex;
        align-items: center;
        gap: 13px;
        margin-bottom: 16px;
    }


    .section-icon {
        width: 44px;
        height: 44px;

        display: flex;
        align-items: center;
        justify-content: center;

        border-radius: 13px;

        background:
            linear-gradient(
                135deg,
                rgba(37, 99, 235, 0.25),
                rgba(6, 182, 212, 0.12)
            );

        border:
            1px solid
            rgba(96, 165, 250, 0.2);

        color: #60a5fa;
        font-size: 18px;
    }


    .section-title {
        font-size: 21px;
        font-weight: 700;
        color: #f8fafc;
    }


    .section-description {
        color: #64748b;
        font-size: 13px;
        margin-top: 3px;
    }


    /* ======================================================
       GLASS CARD
    ====================================================== */

    .glass-card {
        padding: 26px;

        border-radius: 20px;

        background:
            linear-gradient(
                145deg,
                rgba(15, 23, 42, 0.82),
                rgba(15, 23, 42, 0.58)
            );

        border:
            1px solid
            rgba(148, 163, 184, 0.13);

        backdrop-filter: blur(20px);

        box-shadow:
            0 20px 60px
            rgba(0, 0, 0, 0.25);

        transition:
            all 0.3s ease;
    }


    .glass-card:hover {
        transform: translateY(-3px);

        border-color:
            rgba(96, 165, 250, 0.24);

        box-shadow:
            0 25px 70px
            rgba(0, 0, 0, 0.32);
    }


    /* ======================================================
       INPUTS
    ====================================================== */

    div[data-baseweb="select"] > div {
        background:
            rgba(15, 23, 42, 0.9) !important;

        border:
            1px solid
            rgba(148, 163, 184, 0.18) !important;

        border-radius: 11px !important;
    }


    div[data-baseweb="input"] > div {
        background:
            rgba(15, 23, 42, 0.9) !important;

        border:
            1px solid
            rgba(148, 163, 184, 0.18) !important;

        border-radius: 11px !important;
    }


    /* ======================================================
       BUTTON
    ====================================================== */

    .stButton {
        margin-top: 22px;
    }


    .stButton > button {
        width: 100%;
        height: 60px;

        border: none;
        border-radius: 15px;

        background:
            linear-gradient(
                90deg,
                #2563eb,
                #0891b2,
                #2563eb
            );

        background-size: 200% auto;

        color: white;

        font-size: 15px;
        font-weight: 800;
        letter-spacing: 0.5px;

        box-shadow:
            0 10px 35px
            rgba(37, 99, 235, 0.3);

        animation:
            buttonFlow
            4s
            linear
            infinite;
    }


    @keyframes buttonFlow {
        0% {
            background-position: 0% center;
        }

        100% {
            background-position: 200% center;
        }
    }


    .stButton > button:hover {
        transform:
            translateY(-3px)
            scale(1.01);

        box-shadow:
            0 15px 45px
            rgba(37, 99, 235, 0.45);
    }


    /* ======================================================
       RESULT
    ====================================================== */

    .result-card {
        margin-top: 35px;
        padding: 40px;

        text-align: center;

        border-radius: 25px;

        background:
            linear-gradient(
                145deg,
                rgba(30, 64, 175, 0.22),
                rgba(15, 23, 42, 0.78)
            );

        border:
            1px solid
            rgba(96, 165, 250, 0.25);

        box-shadow:
            0 30px 80px
            rgba(0, 0, 0, 0.35);

        animation:
            resultEnter
            0.8s
            ease-out;
    }


    @keyframes resultEnter {
        from {
            opacity: 0;
            transform:
                translateY(35px)
                scale(0.95);
        }

        to {
            opacity: 1;
            transform:
                translateY(0)
                scale(1);
        }
    }


    .result-label {
        color: #64748b;

        font-size: 11px;

        font-weight: 800;

        letter-spacing: 2px;

        text-transform: uppercase;
    }


    .probability {
        margin: 8px 0;

        font-size: 72px;

        font-weight: 800;

        letter-spacing: -4px;

        background:
            linear-gradient(
                90deg,
                #60a5fa,
                #22d3ee
            );

        -webkit-background-clip: text;
        background-clip: text;

        -webkit-text-fill-color: transparent;
    }


    .risk-text {
        font-size: 21px;
        font-weight: 700;
        color: #f8fafc;
    }


    .risk-bar {
        max-width: 650px;

        height: 11px;

        margin:
            25px auto 10px;

        background:
            rgba(30, 41, 59, 0.8);

        border-radius: 20px;

        overflow: hidden;
    }


    .risk-fill {
        height: 100%;

        border-radius: 20px;

        background:
            linear-gradient(
                90deg,
                #22c55e,
                #eab308,
                #ef4444
            );

        animation:
            fillBar
            1.5s
            ease-out;
    }


    @keyframes fillBar {
        from {
            width: 0%;
        }
    }


    .threshold {
        color: #64748b;
        font-size: 12px;
        margin-top: 10px;
    }


    /* ======================================================
       STATS
    ====================================================== */

    .stats-row {
        display: grid;

        grid-template-columns:
            repeat(3, 1fr);

        gap: 15px;

        margin-top: 22px;
    }


    .stat {
        padding: 20px;

        text-align: center;

        border-radius: 16px;

        background:
            rgba(15, 23, 42, 0.65);

        border:
            1px solid
            rgba(148, 163, 184, 0.1);
    }


    .stat-value {
        font-size: 23px;
        font-weight: 800;
        color: #60a5fa;
    }


    .stat-label {
        margin-top: 5px;

        color: #64748b;

        font-size: 11px;

        text-transform: uppercase;

        letter-spacing: 0.7px;
    }


    /* ======================================================
       FOOTER
    ====================================================== */

    .footer {
        margin-top: 70px;

        padding-top: 25px;

        text-align: center;

        border-top:
            1px solid
            rgba(148, 163, 184, 0.1);

        color: #475569;

        font-size: 12px;
    }


    </style>

    <div class="orb orb-one"></div>
    <div class="orb orb-two"></div>
    <div class="orb orb-three"></div>
    """)
)


# ============================================================
# HERO
# ============================================================

st.html("""
<div class="hero">

    <div class="eyebrow">
        <span class="status-dot"></span>
        AI CUSTOMER INTELLIGENCE
    </div>

    <h1 class="hero-title">
        Predict customer
        <span class="gradient-text">churn</span>
        before it happens.
    </h1>

    <div class="hero-description">
        An intelligent machine learning system that analyzes
        customer behavior, services, contracts and billing
        information to estimate churn probability in real time.
    </div>

    <div class="badge-row">

        <div class="badge">
            ◈ XGBoost
        </div>

        <div class="badge">
            ◉ 84.25% ROC-AUC
        </div>

        <div class="badge">
            ⚡ Real-Time Prediction
        </div>

        <div class="badge">
            ◇ Intelligent Analytics
        </div>

    </div>

</div>
""")


# ============================================================
# CUSTOMER PROFILE HEADER
# ============================================================

st.html("""
<div class="section">

    <div class="section-header">

        <div class="section-icon">
            ◉
        </div>

        <div>

            <div class="section-title">
                Customer Profile
            </div>

            <div class="section-description">
                Basic customer information
            </div>

        </div>

    </div>

    <div class="glass-card">
""")


# ============================================================
# CUSTOMER PROFILE INPUTS
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1],
        format_func=lambda x:
            "Yes" if x == 1 else "No"
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )


with col2:

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


with col3:

    multiple_lines = st.selectbox(
        "Multiple Lines",
        [
            "Yes",
            "No",
            "No phone service"
        ]
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


st.html("""
    </div>
</div>
""")


# ============================================================
# SERVICE SECTION
# ============================================================

st.html("""
<div class="section">

    <div class="section-header">

        <div class="section-icon">
            ◇
        </div>

        <div>

            <div class="section-title">
                Service Usage
            </div>

            <div class="section-description">
                Customer subscriptions and service features
            </div>

        </div>

    </div>

    <div class="glass-card">
""")


col1, col2, col3 = st.columns(3)


with col1:

    internet_service = st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )

    online_security = st.selectbox(
        "Online Security",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    online_backup = st.selectbox(
        "Online Backup",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )


with col2:

    device_protection = st.selectbox(
        "Device Protection",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    tech_support = st.selectbox(
        "Tech Support",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )


with col3:

    streaming_movies = st.selectbox(
        "Streaming Movies",
        [
            "Yes",
            "No",
            "No internet service"
        ]
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


st.html("""
    </div>
</div>
""")


# ============================================================
# BILLING
# ============================================================

st.html("""
<div class="section">

    <div class="section-header">

        <div class="section-icon">
            $
        </div>

        <div>

            <div class="section-title">
                Billing Information
            </div>

            <div class="section-description">
                Customer payment and billing information
            </div>

        </div>

    </div>

    <div class="glass-card">
""")


col1, col2 = st.columns(2)


with col1:

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0,
        step=1.0
    )


with col2:

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=840.0,
        step=10.0
    )


st.html("""
    </div>
</div>
""")


# ============================================================
# PREDICT
# ============================================================

predict_button = st.button(
    "◈  ANALYZE CUSTOMER CHURN",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    customer = pd.DataFrame(
        [
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
        ]
    )


    customer_processed = preprocessor.transform(
        customer
    )


    churn_probability = model.predict_proba(
        customer_processed
    )[0][1]


    prediction = (
        churn_probability >= threshold
    )


    percentage = churn_probability * 100


    if churn_probability >= 0.70:

        risk = "HIGH RISK"

    elif churn_probability >= 0.40:

        risk = "MEDIUM RISK"

    else:

        risk = "LOW RISK"


    # ========================================================
    # RESULT
    # ========================================================

    st.html(
        f"""
        <div class="result-card">

            <div class="result-label">
                AI PREDICTION RESULT
            </div>

            <div class="probability">
                {percentage:.2f}%
            </div>

            <div class="risk-text">
                ● {risk}
            </div>

            <div class="risk-bar">

                <div
                    class="risk-fill"
                    style="width: {percentage:.2f}%"
                ></div>

            </div>

            <div class="threshold">
                Classification threshold:
                <strong>{threshold:.2f}</strong>
            </div>

        </div>
        """
    )


    if prediction:

        st.error(
            "⚠️ The model predicts that this customer is likely to churn."
        )

    else:

        st.success(
            "✓ The model predicts that this customer is likely to stay."
        )


    # ========================================================
    # STATISTICS
    # ========================================================

    st.html(
        f"""
        <div class="stats-row">

            <div class="stat">

                <div class="stat-value">
                    {percentage:.1f}%
                </div>

                <div class="stat-label">
                    Churn Probability
                </div>

            </div>


            <div class="stat">

                <div class="stat-value">
                    {threshold:.2f}
                </div>

                <div class="stat-label">
                    Decision Threshold
                </div>

            </div>


            <div class="stat">

                <div class="stat-value">
                    XGBoost
                </div>

                <div class="stat-label">
                    Prediction Model
                </div>

            </div>

        </div>
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.html("""
<div class="footer">

    <strong>CHURNIQ</strong>
    &nbsp; • &nbsp;
    AI-Powered Customer Churn Prediction

    <br><br>

    Built with Python
    • XGBoost
    • Scikit-learn
    • Streamlit

</div>
""")