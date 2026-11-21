"""Offline tests for the input validation in src/predict.py.

Run from the repo root:  python -m pytest tests/
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from predict import REQUIRED_COLUMNS, SAMPLE_CUSTOMER, validate_customer  # noqa: E402


def test_sample_customer_has_every_required_column():
    assert set(SAMPLE_CUSTOMER.columns) == set(REQUIRED_COLUMNS)


def test_missing_column_is_named():
    row = SAMPLE_CUSTOMER.drop(columns=["Contract"])
    with pytest.raises(ValueError, match="Contract"):
        validate_customer(row)


def test_several_missing_columns_are_all_named():
    row = SAMPLE_CUSTOMER.drop(columns=["Contract", "tenure"])
    with pytest.raises(ValueError) as error:
        validate_customer(row)
    assert "Contract" in str(error.value)
    assert "tenure" in str(error.value)


def test_extra_columns_are_allowed():
    row = SAMPLE_CUSTOMER.copy()
    row["favourite_colour"] = "blue"
    validate_customer(row)  # must not raise


def test_more_than_one_row_is_refused():
    two_rows = pd.concat([SAMPLE_CUSTOMER, SAMPLE_CUSTOMER], ignore_index=True)
    with pytest.raises(ValueError, match="exactly one"):
        validate_customer(two_rows)
