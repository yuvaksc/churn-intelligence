"""churn_ml.predictor — how the app gets a churn score for one customer.

    Predictor            the contract: anything with predict(customer) → score + label.
                         Agent 1 depends on this, so tests can pass a fake model.
    PipelinePredictor    the scikit-learn model: loads models/champion.joblib (or
                         MLFLOW_MODEL_URI, e.g. models:/churn-xgb@champion) and returns the
                         calibrated probability, HIGH/LOW at the economic threshold.
                         Local, Docker, training and batch scoring.
    OnnxPredictor        the same model exported to models/champion.onnx (MODEL_FORMAT=onnx,
                         set on Vercel): onnxruntime + numpy only. The scikit-learn stack
                         (scikit-learn, scipy, pandas, xgboost) is ~380 MB, more than a
                         Vercel function may hold. Scores match PipelinePredictor.
    get_predictor()      builds the right one once per process.

The request path never computes SHAP: drivers are precomputed for every customer by
`churn_ml.scoring` and read from the database.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Protocol, TypedDict, runtime_checkable

import numpy as np


class ShapDriver(TypedDict):
    feature: str
    shap_value: float
    raw_value: float
    direction: str


class Prediction(TypedDict):
    risk_score: float          # calibrated churn probability, 0..1
    risk_label: str            # "HIGH" | "LOW"
    threshold: float           # decision threshold the label was cut at
    shap_drivers: list[ShapDriver]


@runtime_checkable
class Predictor(Protocol):
    @property
    def model_version(self) -> str: ...

    def predict(self, customer_raw: dict) -> Prediction: ...


# Default is anchored to the repo root, not the working directory (a serverless
# function's working directory is not guaranteed to be the repo root).
BUNDLE_PATH = Path(os.getenv("MODEL_BUNDLE_PATH")
                   or Path(__file__).resolve().parents[3] / "models" / "champion.joblib")


class PipelinePredictor:
    """The real Predictor: the calibrated sklearn Pipeline from models/champion.joblib."""

    def __init__(self, model, threshold: float, version: str):
        self.model, self.threshold, self._version = model, float(threshold), version

    @property
    def model_version(self) -> str:
        return self._version

    @classmethod
    def from_bundle(cls, path: Path = BUNDLE_PATH) -> "PipelinePredictor":
        import joblib
        b = joblib.load(path)
        return cls(b["model"], b["threshold"], b["version"])

    @classmethod
    def from_mlflow(cls, uri: str) -> "PipelinePredictor":
        import mlflow
        from churn_ml.registry import MODEL_NAME, _mlflow
        _mlflow()
        model = mlflow.sklearn.load_model(uri)
        client = mlflow.MlflowClient()
        alias = uri.rsplit("@", 1)[-1] if "@" in uri else None
        mv = client.get_model_version_by_alias(MODEL_NAME, alias) if alias else None
        threshold = float(mv.tags.get("threshold", 0.5)) if mv else 0.5
        return cls(model, threshold, f"{MODEL_NAME}:v{mv.version}" if mv else uri)

    def predict_proba(self, rows: list[dict]) -> np.ndarray:
        from churn_ml.model import to_frame
        return self.model.predict_proba(to_frame(rows))[:, 1]

    def predict(self, customer_raw: dict) -> dict:
        return _prediction(float(self.predict_proba([customer_raw])[0]), self.threshold)


def _prediction(p: float, threshold: float) -> dict:
    return {
        "risk_score": round(p, 4),
        "risk_label": "HIGH" if p >= threshold else "LOW",
        "threshold": round(threshold, 4),
        "shap_drivers": [],          # precomputed by churn_ml.scoring, read from the database
    }


ONNX_PATH = BUNDLE_PATH.with_suffix(".onnx")
_SERVICES = ["Online Security", "Online Backup", "Device Protection",
             "Tech Support", "Streaming TV", "Streaming Movies"]


def _number(value) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if f != f else f              # NaN → 0, as churn_ml.model.to_frame does


class OnnxPredictor:
    """The champion from models/champion.onnx (written by churn_ml.registry.export_onnx).

    The ONNX graph holds the XGBoost trees. Feature engineering, encoding and isotonic
    calibration are the fitted values stored in the file's metadata, applied here exactly
    as churn_data.features.FeatureEngineer and churn_ml.model.preprocessor do."""

    def __init__(self, path: Path = ONNX_PATH):
        import json

        import onnxruntime as ort
        self._session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        meta = self._session.get_modelmeta().custom_metadata_map
        self._spec = json.loads(meta["churn_spec"])
        self.threshold, self._version = float(meta["threshold"]), meta["version"]
        self._iso_x = np.asarray(self._spec["isotonic_x"])
        self._iso_y = np.asarray(self._spec["isotonic_y"])

    @property
    def model_version(self) -> str:
        return self._version

    def features(self, raw: dict) -> list[float]:
        """One customer → the encoded feature vector the XGBoost trees were trained on."""
        s = self._spec
        tenure, monthly = _number(raw.get("Tenure Months")), _number(raw.get("Monthly Charges"))
        total = _number(raw.get("Total Charges"))
        bundle = f"{raw.get('Internet Service')}|{raw.get('Phone Service')}"
        engineered = {
            "Charge Ratio": monthly / (total + 1e-9),
            "Avg Monthly Charge": total / (tenure + 1e-9),
            "Services Count": sum(raw.get(c) == "Yes" for c in _SERVICES),
            "High Risk Flag": int(raw.get("Contract") == "Month-to-month" and monthly > s["median_charge"]),
            "Tenure Bin": 0 if tenure <= 12 else 1 if tenure <= 36 else 2 if tenure <= 60 else 3,
            "Sticky Payment": int(raw.get("Paperless Billing", "No") == "Yes"
                                  and "automatic" in str(raw.get("Payment Method", "")).lower()),
            "Charge Index": (monthly / (s["bundle_medians"].get(bundle, s["median_charge"]) + 1e-9)
                             if s["bundle_medians"] else 1.0),
        }
        numbers = {"Tenure Months": tenure, "Monthly Charges": monthly, "Total Charges": total,
                   **engineered}
        contract = raw.get("Contract")
        row = [float(s["contract"].index(contract)) if contract in s["contract"] else -1.0]
        for col, kept in s["onehot"]:
            row += [1.0 if str(raw.get(col)) == cat else 0.0 for cat in kept]
        return row + [float(numbers[col]) for col in s["numeric"]]

    def predict_proba(self, rows: list[dict]) -> np.ndarray:
        X = np.asarray([self.features(r) for r in rows], dtype=np.float32)
        probs = self._session.run(None, {"input": X})[1]
        raw = np.asarray([p[1] for p in probs] if isinstance(probs, list) else probs[:, 1],
                         dtype=float)
        return np.interp(raw, self._iso_x, self._iso_y)      # isotonic calibration (clipped)

    def predict(self, customer_raw: dict) -> dict:
        return _prediction(float(self.predict_proba([customer_raw])[0]), self.threshold)


@lru_cache(maxsize=1)
def get_predictor():
    if os.getenv("MODEL_FORMAT") == "onnx":
        return OnnxPredictor(ONNX_PATH)
    uri = os.getenv("MLFLOW_MODEL_URI")
    if uri:
        return PipelinePredictor.from_mlflow(uri)
    if BUNDLE_PATH.exists():
        return PipelinePredictor.from_bundle(BUNDLE_PATH)
    raise RuntimeError(f"No champion model at {BUNDLE_PATH}: run `python -m churn_ml.retrain --promote force`")
