"""Offline tests for the decision threshold in src/predict.py.

These use stub preprocessor/model objects, so no pickle or training data
is needed.

Run from the repo root:  python -m pytest tests/
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from predict import REQUIRED_COLUMNS, predict_customer  # noqa: E402


class PassthroughPreprocessor:
    def transform(self, customer):
        return customer


class FixedProbabilityModel:
    def __init__(self, churn_probability):
        self.churn_probability = churn_probability

    def predict_proba(self, _customer):
        return np.array([[1 - self.churn_probability, self.churn_probability]])


def _model_data(probability, threshold):
    return {
        "preprocessor": PassthroughPreprocessor(),
        "model": FixedProbabilityModel(probability),
        "threshold": threshold,
    }


def _row():
    return pd.DataFrame([{column: "x" for column in REQUIRED_COLUMNS}])


def test_probability_below_threshold_predicts_no_churn():
    probability, prediction = predict_customer(
        _row(), _model_data(0.39, threshold=0.4))
    assert probability == pytest.approx(0.39)
    assert prediction is False


def test_probability_at_threshold_predicts_churn():
    probability, prediction = predict_customer(
        _row(), _model_data(0.4, threshold=0.4))
    assert probability == pytest.approx(0.4)
    assert prediction is True


def test_probability_above_threshold_predicts_churn():
    probability, prediction = predict_customer(
        _row(), _model_data(0.9, threshold=0.4))
    assert probability == pytest.approx(0.9)
    assert prediction is True
