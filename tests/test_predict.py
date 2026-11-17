"""Offline tests for src/predict.py using the saved model pickle.

Run from the repo root:  python -m pytest tests/
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from predict import SAMPLE_CUSTOMER, load_model, predict_customer  # noqa: E402


@pytest.fixture(scope="module")
def model_data():
    return load_model()


def _customer(**overrides):
    row = SAMPLE_CUSTOMER.iloc[0].to_dict()
    row.update(overrides)
    return pd.DataFrame([row])


def test_load_model_has_expected_keys(model_data):
    assert set(model_data) == {"preprocessor", "model", "threshold"}
    assert 0 < model_data["threshold"] < 1


def test_predict_customer_returns_probability_and_label(model_data):
    probability, prediction = predict_customer(
        SAMPLE_CUSTOMER, model_data
    )
    assert 0 <= probability <= 1
    assert isinstance(prediction, bool)


def test_risky_customer_scores_higher_than_loyal_one(model_data):
    risky = _customer()  # month-to-month, electronic check, 12 months
    loyal = _customer(
        tenure=70,
        Contract="Two year",
        PaymentMethod="Credit card (automatic)",
        TotalCharges=5600.0,
    )

    risky_probability, _ = predict_customer(risky, model_data)
    loyal_probability, _ = predict_customer(loyal, model_data)

    assert risky_probability > loyal_probability
