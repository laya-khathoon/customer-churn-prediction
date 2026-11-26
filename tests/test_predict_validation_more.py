"""More offline tests for validate_customer."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from predict import SAMPLE_CUSTOMER, validate_customer  # noqa: E402


def test_extra_columns_are_allowed():
    row = SAMPLE_CUSTOMER.copy()
    row["Notes"] = "not used by the model"
    validate_customer(row)


def test_two_rows_are_refused():
    two = SAMPLE_CUSTOMER._append(SAMPLE_CUSTOMER, ignore_index=True)
    with pytest.raises(ValueError, match="exactly one"):
        validate_customer(two)


def test_empty_frame_is_refused():
    empty = SAMPLE_CUSTOMER.iloc[0:0]
    with pytest.raises(ValueError):
        validate_customer(empty)


def test_sample_customer_passes_validation():
    validate_customer(SAMPLE_CUSTOMER)
