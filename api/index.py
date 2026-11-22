"""Stateless Vercel adapter. Original Streamlit entrypoint remains unchanged.

Predictions run with only the Python standard library: the trained
XGBoost library alone is larger than Vercel's function size limit, so
scripts/export_web_model.py converts models/churn_model.pkl into
api/model_web.json and web_model.py evaluates it dependency-free.
"""

from http.server import BaseHTTPRequestHandler
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from web_model import load_web_model, predict_customer, REQUIRED_FIELDS

WEB_MODEL = load_web_model()


def run_action(data):
    action = data.get("action", "predict")
    if action == "health":
        return {
            "status": "ok",
            "model": WEB_MODEL["name"],
            "threshold": WEB_MODEL["threshold"],
            "trees": len(WEB_MODEL["trees"]),
        }
    if action != "predict":
        raise ValueError("Unknown action.")

    customer = data.get("customer")
    if not isinstance(customer, dict):
        raise ValueError("Missing customer object.")

    probability, prediction = predict_customer(customer, WEB_MODEL)
    percentage = probability * 100
    if probability >= 0.70:
        risk = "HIGH RISK"
    elif probability >= 0.40:
        risk = "MEDIUM RISK"
    else:
        risk = "LOW RISK"

    return {
        "probability": round(probability, 6),
        "percentage": round(percentage, 2),
        "prediction": prediction,
        "risk": risk,
        "threshold": WEB_MODEL["threshold"],
        "verdict": (
            "The model predicts that this customer is likely to churn."
            if prediction
            else "The model predicts that this customer is likely to stay."
        ),
        "fields_used": len(REQUIRED_FIELDS),
    }


class handler(BaseHTTPRequestHandler):
    def _reply(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._reply(200, run_action({"action": "health"}))

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length > 65536:
                self._reply(413, {"error": "Request too large."})
                return
            data = json.loads(self.rfile.read(length) or b"{}")
            self._reply(200, run_action(data))
        except ValueError as error:
            self._reply(400, {"error": str(error)})
        except Exception:
            self._reply(500, {"error": "Prediction failed. Try again."})
