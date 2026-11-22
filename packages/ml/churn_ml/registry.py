"""churn_ml.registry — MLflow tracking + model registry (champion / challenger).

    log_candidate(result)   → run with params, metrics, model card;
                               registers a new version of `churn-xgb`, alias @challenger
    compare(result)         → re-scores the current @champion on the SAME test split
    promote(version)        → moves @champion, exports models/champion.joblib for serving
                               (local / Docker) and models/champion.onnx (Vercel)

Promotion rule: the challenger must match or beat the champion's test ROC-AUC and
must not worsen the Brier score by more than BRIER_TOLERANCE (calibration guard).

Tracking store: MLFLOW_TRACKING_URI, default a local SQLite file under mlruns/.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import joblib

MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "churn-xgb")
BRIER_TOLERANCE = 0.005
AUC_TOLERANCE = 0.0
BUNDLE_PATH = Path(os.getenv("MODEL_BUNDLE_PATH", "models/champion.joblib"))


def _mlflow():
    import mlflow
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        Path("mlruns").mkdir(exist_ok=True)
        uri = "sqlite:///mlruns/mlflow.db"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("churn-retraining")
    return mlflow


def log_candidate(result) -> dict:
    """Log the training run and register it as the new @challenger."""
    from mlflow.models import infer_signature
    from churn_ml.training import model_card

    mlflow = _mlflow()
    client = mlflow.MlflowClient()
    with mlflow.start_run(run_name=f"train-{result.version}") as run:
        mlflow.log_params(result.params)
        mlflow.log_param("data_fingerprint", result.data_fingerprint)
        mlflow.log_metrics({f"test_{k}": v for k, v in result.metrics.items()})
        mlflow.log_metrics({f"baseline_{k}": v for k, v in result.baseline_metrics.items()})
        with tempfile.TemporaryDirectory() as d:
            Path(d, "model_card.md").write_text(model_card(result), encoding="utf-8")
            mlflow.log_artifacts(d)
        sample = result.splits.X_test.head(50)
        info = mlflow.sklearn.log_model(
            sk_model=result.model, name="model", registered_model_name=MODEL_NAME,
            signature=infer_signature(sample, result.model.predict_proba(sample)[:, 1]),
            input_example=sample.head(3),
        )
        run_id = run.info.run_id

    version = str(info.registered_model_version)
    client.set_model_version_tag(MODEL_NAME, version, "threshold", str(result.threshold))
    client.set_model_version_tag(MODEL_NAME, version, "build", result.version)
    client.set_registered_model_alias(MODEL_NAME, "challenger", version)
    return {"run_id": run_id, "version": version}


def champion_version():
    mlflow = _mlflow()
    try:
        return mlflow.MlflowClient().get_model_version_by_alias(MODEL_NAME, "champion")
    except Exception:
        return None


def compare(result) -> dict:
    """Challenger vs champion on the challenger's held-out test split."""
    from churn_ml.training import evaluate

    champ = champion_version()
    if champ is None:
        return {"champion": None, "promote": True, "reason": "no champion yet"}

    mlflow = _mlflow()
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
    threshold = float(champ.tags.get("threshold", 0.5))
    s = result.splits
    champ_metrics = evaluate(model.predict_proba(s.X_test)[:, 1], s.y_test,
                             s.X_test["Monthly Charges"], threshold)
    new = result.metrics
    auc_ok = new["roc_auc"] >= champ_metrics["roc_auc"] - AUC_TOLERANCE
    brier_ok = new["brier"] <= champ_metrics["brier"] + BRIER_TOLERANCE
    return {
        "champion": {"version": champ.version, **champ_metrics},
        "promote": bool(auc_ok and brier_ok),
        "reason": ("challenger matches or beats champion" if auc_ok and brier_ok else
                   "ROC-AUC below champion" if not auc_ok else "calibration (Brier) regressed"),
    }


def promote(version: str, result=None) -> None:
    mlflow = _mlflow()
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(MODEL_NAME, "champion", version)
    mv = client.get_model_version(MODEL_NAME, version)
    model = result.model if result is not None else mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{version}")
    export_bundle(model, float(mv.tags.get("threshold", 0.5)), f"{MODEL_NAME}:v{version}",
                  result.metrics if result is not None else {})


def export_bundle(model, threshold: float, version: str, metrics: dict | None = None) -> Path:
    """Self-contained serving artifact: the API loads this without an MLflow server."""
    BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "threshold": threshold, "version": version,
                 "metrics": metrics or {}}, BUNDLE_PATH)
    (BUNDLE_PATH.parent / "champion.json").write_text(
        json.dumps({"version": version, "threshold": threshold, "metrics": metrics or {}}, indent=2),
        encoding="utf-8")
    export_onnx(model, threshold, version)
    return BUNDLE_PATH


def onnx_spec(model) -> dict:
    """Everything the fitted pipeline learned outside the XGBoost trees: the feature
    engineering medians, the encoders' categories and the isotonic calibration curve.
    churn_ml.predictor.OnnxPredictor applies them with plain Python + numpy."""
    from churn_ml.model import CATEGORICAL, NUMERIC

    cal = model.calibrated_classifiers_[0]
    prep = cal.estimator.named_steps["prep"]
    feats, enc = prep.named_steps["features"], prep.named_steps["encode"]
    onehot = enc.named_transformers_["categorical"]
    drop = onehot.drop_idx_ if onehot.drop_idx_ is not None else [None] * len(CATEGORICAL)
    iso = cal.calibrators[0]
    return {
        "median_charge": feats.median_charge_,
        "bundle_medians": feats.bundle_medians_,
        "contract": [str(c) for c in enc.named_transformers_["contract"].categories_[0]],
        "onehot": [[col, [str(c) for j, c in enumerate(cats) if j != drop[i]]]
                   for i, (col, cats) in enumerate(zip(CATEGORICAL, onehot.categories_))],
        "numeric": NUMERIC,
        "isotonic_x": [float(x) for x in iso.X_thresholds_],
        "isotonic_y": [float(y) for y in iso.y_thresholds_],
    }


ONNX_PATH = BUNDLE_PATH.with_suffix(".onnx")


def export_onnx(model, threshold: float, version: str) -> Path:
    """models/champion.onnx: the XGBoost trees as an ONNX graph (onnxmltools), with the
    preprocessing + calibration spec, threshold and version in its metadata. Lets the
    serverless API score live with onnxruntime + numpy, no scikit-learn/scipy/pandas."""
    from onnxmltools import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType

    spec = onnx_spec(model)
    n_features = 1 + sum(len(kept) for _, kept in spec["onehot"]) + len(spec["numeric"])
    clf = model.calibrated_classifiers_[0].estimator.named_steps["clf"]
    onx = convert_xgboost(clf, initial_types=[("input", FloatTensorType([None, n_features]))],
                          target_opset=15)
    for key, value in {"churn_spec": json.dumps(spec), "threshold": str(threshold),
                       "version": version}.items():
        prop = onx.metadata_props.add()
        prop.key, prop.value = key, value
    ONNX_PATH.write_bytes(onx.SerializeToString())
    return ONNX_PATH
