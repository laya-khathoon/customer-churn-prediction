"""Offline tests for web_model.py, the Vercel web version's predictor.

The web predictor evaluates api/model_web.json with only the standard
library. These tests compare it against the real pipeline in
models/churn_model.pkl on every dataset row, plus edge cases.

Run from the repo root:  python -m pytest tests/test_web_model.py
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import web_model  # noqa: E402
from predict import SAMPLE_CUSTOMER, load_model  # noqa: E402

TOLERANCE = 1e-5


@pytest.fixture(scope="module")
def reference():
    return load_model()


@pytest.fixture(scope="module")
def web():
    return web_model.load_web_model()


def _sample_dict(**overrides):
    row = SAMPLE_CUSTOMER.iloc[0].to_dict()
    row.update(overrides)
    return row


def _reference_probability(customer_dict, reference):
    frame = pd.DataFrame([customer_dict])
    processed = reference["preprocessor"].transform(frame)
    return reference["model"].predict_proba(processed)[:, 1][0]


def test_web_model_matches_pipeline_on_sample(reference, web):
    customer = _sample_dict()
    probability, prediction = web_model.predict_customer(customer, web)
    reference_probability = _reference_probability(customer, reference)
    assert abs(probability - reference_probability) < TOLERANCE
    assert prediction == bool(
        reference_probability >= reference["threshold"]
    )


def test_web_model_matches_pipeline_on_every_dataset_row(reference, web):
    df = pd.read_csv("data/customer_churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    X = df.drop(["customerID", "Churn"], axis=1)
    reference_probabilities = reference["model"].predict_proba(
        reference["preprocessor"].transform(X)
    )[:, 1]
    rows = X.where(pd.notnull(X), None)
    worst = 0.0
    for index in range(len(X)):
        customer = {
            key: (value.item() if hasattr(value, "item") else value)
            for key, value in rows.iloc[index].to_dict().items()
        }
        probability, _ = web_model.predict_customer(customer, web)
        worst = max(worst, abs(probability - reference_probabilities[index]))
    assert worst < TOLERANCE


def test_web_model_handles_unseen_category(reference, web):
    customer = _sample_dict(InternetService="Satellite")
    probability, _ = web_model.predict_customer(customer, web)
    reference_probability = _reference_probability(customer, reference)
    assert abs(probability - reference_probability) < TOLERANCE


def test_web_model_rejects_missing_fields(web):
    with pytest.raises(ValueError, match="Missing customer fields"):
        web_model.predict_customer({"gender": "Male"}, web)


def test_web_model_returns_threshold_and_valid_probability(web):
    _, web_threshold = None, web["threshold"]
    assert 0 < web_threshold < 1
    probability, prediction = web_model.predict_customer(_sample_dict(), web)
    assert 0 <= probability <= 1
    assert prediction == bool(probability >= web_threshold)
