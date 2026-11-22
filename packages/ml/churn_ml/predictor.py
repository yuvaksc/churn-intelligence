"""churn_ml.predictor — how the app gets a churn score for one customer.

    Predictor            the contract: anything with predict(customer) → score + label.
                         Agent 1 depends on this, so tests can pass a fake model.
    PipelinePredictor    the real one: loads models/champion.joblib (or MLFLOW_MODEL_URI,
                         e.g. models:/churn-xgb@champion) and returns the calibrated
                         probability, HIGH/LOW at the economic threshold.
    get_predictor()      loads it once per process.

The request path only computes a probability. SHAP drivers are precomputed for every
customer by `churn_ml.scoring` and read from the database, so scoring a customer never
pays for an explainer.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Protocol, TypedDict, runtime_checkable

import joblib
import numpy as np

from churn_ml.model import to_frame


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
        return self.model.predict_proba(to_frame(rows))[:, 1]

    def predict(self, customer_raw: dict) -> dict:
        p = float(self.predict_proba([customer_raw])[0])
        return {
            "risk_score": round(p, 4),
            "risk_label": "HIGH" if p >= self.threshold else "LOW",
            "threshold": round(self.threshold, 4),
            "shap_drivers": [],          # precomputed by churn_ml.scoring, read from the database
        }


@lru_cache(maxsize=1)
def get_predictor():
    uri = os.getenv("MLFLOW_MODEL_URI")
    if uri:
        return PipelinePredictor.from_mlflow(uri)
    if BUNDLE_PATH.exists():
        return PipelinePredictor.from_bundle(BUNDLE_PATH)
    raise RuntimeError(f"No champion model at {BUNDLE_PATH}: run `python -m churn_ml.retrain --promote force`")
