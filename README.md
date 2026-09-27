# Customer Churn Prediction

A machine learning project that predicts whether a customer is likely to churn based on customer demographics, service usage, contract details, and billing information.

## Project Overview

Customer churn refers to customers discontinuing a service.

This project uses machine learning to:

- Analyze customer information
- Preprocess numerical and categorical features
- Train multiple classification models
- Evaluate model performance
- Predict customer churn probability
- Provide an interactive Streamlit application

## Dataset

The project uses the Telco Customer Churn dataset.

The dataset contains:

- 7,043 customer records
- 19 input features after removing the customer ID
- 1 target variable: `Churn`

Target classes:

- `No` → 0
- `Yes` → 1

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Streamlit
- Joblib
- Jupyter Notebook
- Git & GitHub

## Machine Learning Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Feature Selection
   ↓
Train / Validation / Test Split
   ↓
Data Preprocessing
   ↓
Model Training
   ↓
Threshold Selection
   ↓
Model Evaluation
   ↓
Churn Prediction
   ↓
Streamlit Application