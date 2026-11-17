"""Offline tests for the churn training pipeline in src/train.py.

Run from the repo root:  python -m pytest tests/
Everything runs locally against the bundled CSV; no network needed.
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from train import (  # noqa: E402
    NUMERICAL_FEATURES,
    build_preprocessor,
    categorical_features_for,
    evaluate_model,
    load_data,
    make_splits,
    save_model,
    select_threshold,
    split_features_target,
    train_model,
)


@pytest.fixture(scope="module")
def df():
    return load_data()


@pytest.fixture(scope="module")
def splits(df):
    X, y = split_features_target(df)
    return X, make_splits(X, y)


def test_load_data_drops_id_and_parses_charges(df):
    assert "customerID" not in df.columns
    assert pd.api.types.is_numeric_dtype(df["TotalCharges"])
    assert len(df) > 7000


def test_split_features_target_maps_churn(df):
    X, y = split_features_target(df)
    assert "Churn" not in X.columns
    assert set(y.unique()) == {0, 1}
    assert len(X) == len(y)


def test_categorical_features_exclude_numerical(df):
    X, _ = split_features_target(df)
    categorical = categorical_features_for(X)
    for feature in NUMERICAL_FEATURES:
        assert feature not in categorical
    assert "Contract" in categorical
    assert len(categorical) + len(NUMERICAL_FEATURES) == X.shape[1]


def test_make_splits_shapes_and_stratification(splits):
    X, (X_train, X_val, X_test, y_train, y_val, y_test) = splits
    total = len(X)
    assert len(X_test) == pytest.approx(total * 0.20, abs=2)
    assert len(X_val) == pytest.approx(total * 0.16, abs=2)
    assert len(X_train) == pytest.approx(total * 0.64, abs=2)
    # stratified: churn rate close in every split
    overall = pd.concat([y_train, y_val, y_test]).mean()
    for part in (y_train, y_val, y_test):
        assert part.mean() == pytest.approx(overall, abs=0.01)


def test_full_pipeline_reproduces_expected_metrics(splits):
    X, (X_train, X_val, X_test, y_train, y_val, y_test) = splits

    preprocessor = build_preprocessor(
        NUMERICAL_FEATURES,
        categorical_features_for(X),
    )
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    model = train_model(X_train_processed, y_train)

    threshold, val_f1 = select_threshold(
        model, X_val_processed, y_val
    )
    assert threshold in {0.3, 0.4, 0.5, 0.6, 0.7}
    assert 0.5 < val_f1 < 0.7

    metrics, y_pred, y_probability = evaluate_model(
        model, X_test_processed, y_test, threshold
    )
    # README reports accuracy 0.7814, F1 0.61, ROC-AUC 0.84
    assert metrics["accuracy"] > 0.75
    assert metrics["f1"] > 0.55
    assert metrics["roc_auc"] > 0.80
    assert len(y_pred) == len(y_test)
    assert ((y_probability >= 0) & (y_probability <= 1)).all()


def test_save_model_roundtrip(splits, tmp_path):
    X, (X_train, _, _, y_train, _, _) = splits

    preprocessor = build_preprocessor(
        NUMERICAL_FEATURES,
        categorical_features_for(X),
    )
    X_train_processed = preprocessor.fit_transform(X_train)
    model = train_model(X_train_processed, y_train)

    path = tmp_path / "model.pkl"
    save_model(preprocessor, model, 0.4, path=str(path))

    import joblib
    loaded = joblib.load(path)
    assert set(loaded) == {"preprocessor", "model", "threshold"}
    assert loaded["threshold"] == 0.4
