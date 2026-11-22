"""Export the trained churn model to a dependency-free JSON web model.

Reads models/churn_model.pkl (preprocessor + XGBoost + threshold) and
writes api/model_web.json, which api/index.py can use for predictions
with only the Python standard library. Run from the repo root:

    python scripts/export_web_model.py

The original pickle, training pipeline and Streamlit app are unchanged;
this only adds a small JSON export for the Vercel web version.
"""

import json
import math
import os

import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PICKLE_PATH = os.path.join(ROOT, "models", "churn_model.pkl")
OUTPUT_PATH = os.path.join(ROOT, "api", "model_web.json")


def export():
    model_data = joblib.load(PICKLE_PATH)
    preprocessor = model_data["preprocessor"]
    model = model_data["model"]

    numerical = {"columns": [], "median": [], "mean": [], "scale": []}
    categorical = {"columns": [], "mode": [], "categories": []}

    for name, pipeline, columns in preprocessor.transformers_:
        if name == "num":
            numerical["columns"] = list(columns)
            numerical["median"] = [
                float(v) for v in pipeline.named_steps["imputer"].statistics_
            ]
            numerical["mean"] = [
                float(v) for v in pipeline.named_steps["scaler"].mean_
            ]
            numerical["scale"] = [
                float(v) for v in pipeline.named_steps["scaler"].scale_
            ]
        elif name == "cat":
            categorical["columns"] = list(columns)
            categorical["mode"] = [
                str(v) for v in pipeline.named_steps["imputer"].statistics_
            ]
            categorical["categories"] = [
                [str(option) for option in options]
                for options in pipeline.named_steps["encoder"].categories_
            ]

    booster = model.get_booster()
    config = json.loads(booster.save_config())
    raw_base = config["learner"]["learner_model_param"]["base_score"]
    base_score = float(str(raw_base).strip("[]").split(",")[0])

    trees = [
        json.loads(tree)
        for tree in booster.get_dump(dump_format="json")
    ]

    web_model = {
        "name": "customer-churn-prediction web model",
        "source": "models/churn_model.pkl",
        "objective": "binary:logistic",
        "base_margin": logit(base_score),
        "threshold": float(model_data["threshold"]),
        "numerical": numerical,
        "categorical": categorical,
        "trees": trees,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as handle:
        json.dump(web_model, handle, separators=(",", ":"))
    print(f"Wrote {OUTPUT_PATH} with {len(trees)} trees.")


def logit(probability):
    return math.log(probability / (1 - probability))


if __name__ == "__main__":
    export()
