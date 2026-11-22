# Model Card: Customer Churn Prediction

## Intended use

Educational machine-learning project that estimates the probability that a
Telco customer will churn, and demonstrates the full workflow from raw data
to a deployed Streamlit app. It is not used for real business decisions.

## Data

- Telco Customer Churn dataset: 7,043 customer records.
- 19 input features after dropping `customerID`; the target is `Churn`
  (`Yes` -> 1, `No` -> 0).
- Split: 64% train, 16% validation, 20% test, stratified with
  `random_state=42`.

## Model

- XGBoost classifier inside a scikit-learn preprocessing pipeline
  (numerical and categorical features preprocessed together).
- The decision threshold is chosen on the validation set as the best F1
  between 0.3 and 0.7, and is 0.4 for the saved model.
- Saved artifact: `models/churn_model.pkl` (preprocessor, model, threshold).

## Test metrics (1,409 customers)

| Metric | Score |
| --- | --- |
| Accuracy | 0.7814 |
| Precision (churn) | 0.5797 |
| Recall (churn) | 0.6417 |
| F1 (churn) | 0.6091 |
| ROC-AUC | 0.8425 |

Confusion matrix (rows actual, columns predicted, order No Churn / Churn):

```text
[[861 174]
 [134 240]]
```

Re-running `python src/train.py` with newer library versions gives almost
the same numbers (accuracy 0.7814, F1 0.6101, ROC-AUC 0.8423).

## Limitations

- Trained on one public dataset from one telecom provider; it does not
  generalize to other companies or industries.
- The threshold favours catching churners over precision, so about 4 in 10
  churn flags are false alarms on the test set.
- Features are a snapshot in time; the model has no notion of trends over a
  customer's lifetime.

## Reproduce

```bash
pip install -r requirements.txt
python src/train.py     # trains, picks the threshold, writes the pickle
python -m pytest tests/ # offline checks, including threshold behaviour
```
