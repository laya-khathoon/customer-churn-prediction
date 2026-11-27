import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from train import make_splits, split_features_target  # noqa: E402


def _frame(rows=100):
    return pd.DataFrame({
        "tenure": range(rows),
        "Churn": ["Yes" if i % 4 == 0 else "No" for i in range(rows)],
    })


def test_target_is_mapped_to_zero_and_one_and_removed_from_features():
    X, y = split_features_target(_frame(8))
    assert "Churn" not in X.columns
    assert set(y.unique()) == {0, 1}
    assert y.iloc[0] == 1


def test_split_sizes_are_64_16_20_percent():
    X, y = split_features_target(_frame(100))
    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(X, y)
    assert (len(X_train), len(X_val), len(X_test)) == (64, 16, 20)


def test_each_split_keeps_the_churn_rate():
    X, y = split_features_target(_frame(100))
    *_, y_train, y_val, y_test = make_splits(X, y)
    for part in (y_train, y_val, y_test):
        assert abs(part.mean() - 0.25) < 0.05
