"""Dependency-free churn prediction using api/model_web.json.

This mirrors src/predict.py for the Vercel web version. It needs only
the Python standard library, so the serverless function stays far below
Vercel's size limit (the trained XGBoost library alone is larger than
that limit, which is why the web version uses the JSON export instead).

The JSON export is produced by scripts/export_web_model.py from the
same models/churn_model.pkl the Streamlit app uses.
"""

import json
import math
import os
import struct


def _float32(value):
    """Round a float the way XGBoost's float32 internals do."""
    return struct.unpack("f", struct.pack("f", value))[0]

MODEL_WEB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "api", "model_web.json"
)

REQUIRED_FIELDS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]


def load_web_model(path=MODEL_WEB_PATH):
    """Load the JSON export produced by scripts/export_web_model.py."""
    with open(path) as handle:
        return json.load(handle)


def build_features(customer, web_model):
    """Impute, scale and one-hot encode one customer into a float list."""
    numerical = web_model["numerical"]
    features = []
    for position, column in enumerate(numerical["columns"]):
        value = customer.get(column)
        try:
            value = float(value)
            if math.isnan(value):
                value = numerical["median"][position]
        except (TypeError, ValueError):
            value = numerical["median"][position]
        features.append(
            (value - numerical["mean"][position])
            / numerical["scale"][position]
        )

    categorical = web_model["categorical"]
    for position, column in enumerate(categorical["columns"]):
        value = customer.get(column)
        if value is None or value == "":
            value = categorical["mode"][position]
        else:
            value = str(value)
        for option in categorical["categories"][position]:
            features.append(1.0 if value == option else 0.0)
    return features


def _tree_leaf(node, features):
    while "leaf" not in node:
        column = int(node["split"][1:])
        if _float32(features[column]) < _float32(node["split_condition"]):
            next_id = node["yes"]
        else:
            next_id = node["no"]
        node = next(
            child for child in node["children"]
            if child["nodeid"] == next_id
        )
    return node["leaf"]


def predict_customer(customer, web_model):
    """Return (churn_probability, prediction) for one customer dict."""
    missing = [field for field in REQUIRED_FIELDS if field not in customer]
    if missing:
        raise ValueError("Missing customer fields: " + ", ".join(missing))

    features = build_features(customer, web_model)
    margin = web_model["base_margin"] + sum(
        _tree_leaf(tree, features) for tree in web_model["trees"]
    )
    probability = 1.0 / (1.0 + math.exp(-margin))
    return probability, bool(probability >= web_model["threshold"])
