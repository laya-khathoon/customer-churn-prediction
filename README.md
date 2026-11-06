# Customer Churn Prediction

**Live demo:** https://customer-churn-prediction-q5kxzl9kzqgf6putmyc6yt.streamlit.app/

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
```

## Results

Model: XGBoost inside a scikit-learn preprocessing pipeline. Data split: 64% train, 16% validation, 20% test (stratified, `random_state=42`). The decision threshold is picked on the validation set (best F1 among 0.3 to 0.7) and comes out at 0.4.

Test set results (1,409 customers), reproduced by running `python src/train.py`:

| Metric | Score |
| --- | --- |
| Accuracy | 0.7814 |
| Precision (churn) | 0.5793 |
| Recall (churn) | 0.6444 |
| F1 (churn) | 0.6101 |
| ROC-AUC | 0.8423 |

Confusion matrix (rows are actual, columns are predicted, order No Churn / Churn):

```text
[[860 175]
 [133 241]]
```

Exact numbers can move slightly with different library versions.

## Setup and Run

```bash
git clone https://github.com/laya-khathoon/customer-churn-prediction.git
cd customer-churn-prediction
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Train the model and print the metrics (writes `models/churn_model.pkl`):

```bash
python src/train.py
```

Predict for a sample customer from the command line:

```bash
python src/predict.py
```

Run the Streamlit app:

```bash
streamlit run app.py
```

If `models/churn_model.pkl` is missing or was saved with a different library version, the app retrains it once on startup.

## Project Structure

```text
app.py              Streamlit app
src/train.py        Training and evaluation
src/predict.py      Sample prediction script
data/               Telco customer churn CSV
models/             Saved model and preprocessing pipeline
notebooks/          Data exploration
requirements.txt    Direct dependencies
```

## Deployment

The app is deployed on Streamlit Community Cloud from the `main` branch with `app.py` as the entry file.
