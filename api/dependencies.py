"""
api/dependencies.py — Shared application state loaded once at startup.

AWS (Lambda):
  On cold start, queries the SageMaker Model Registry for the latest Approved
  version of the 'churn-xgboost' model group. Reads S3 paths and threshold
  from the version's CustomerMetadataProperties, downloads the pkl files to
  /tmp (cached for warm container reuse), then loads them with joblib.

  Registry metadata keys (verified):
    s3_pipeline_pkl   → xgb_pipeline.pkl   (scoring, 35 features)
    s3_shap_pkl       → xgboost_churn.pkl  (SHAP, 36 features)
    s3_encoders_pkl   → encoders.pkl
    s3_features_35    → feature_names_35.pkl
    s3_features_36    → feature_names_36.pkl
    threshold         → "0.67"  (plain string value, not an S3 path)
    f1_score          → "0.6297"
    roc_auc           → "0.8485"
    encoding_version  → "v1"

  shap_explainer.pkl and shap_values_test.csv are NOT in the registry —
  they are baked into the image and loaded from the image filesystem.

Local (non-Lambda):
  Falls back to loading directly from models/ on disk. No registry or S3 calls.
  Run the system normally with docker compose or uvicorn.
"""

import os
import sys
import boto3
import joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.loader import (
    load_raw, clean, engineer_features,
    encode_for_gbm, three_way_split,
)

MODELS_DIR  = Path("models")
TMP_DIR     = Path("/tmp/churn_models")

MODEL_GROUP = os.environ.get("SAGEMAKER_MODEL_GROUP", "churn-xgboost")
REGION      = os.environ.get("BEDROCK_REGION", "us-east-1")

# Maps logical name → CustomerMetadataProperties key in the registry.
# Verified against actual registry metadata on 2026-06-02.
# Note: threshold is a plain string value in metadata, not an S3 path —
# it is read directly from registry_meta["threshold"], not downloaded.
# Note: s3_shap_pkl points to xgboost_churn.pkl (standalone XGBClassifier
# used by shap_explainer.pkl at inference time) — shap_explainer.pkl itself
# is not registered and is loaded from the baked image.
_META_KEYS = {
    "pipeline": "s3_pipeline_pkl",   # xgb_pipeline.pkl  (SMOTEENN + XGBClassifier, 35 feat)
    "xgboost":  "s3_shap_pkl",       # xgboost_churn.pkl (standalone, 36 feat, used for SHAP)
    "encoders": "s3_encoders_pkl",   # encoders.pkl
    "feat_35":  "s3_features_35",    # feature_names_35.pkl
    "feat_36":  "s3_features_36",    # feature_names_36.pkl
}


@dataclass
class AppState:
    eng_test_raw:  pd.DataFrame          # pre-encoding test rows — passed to war room agents
    X_test:        pd.DataFrame          # encoded test rows — used for pre-scoring
    y_test:        pd.Series             # ground truth labels
    risk_scores:   np.ndarray            # pre-computed predict_proba for all test customers
    shap_df:       pd.DataFrame          # SHAP values (1409 × 36) from shap_values_test.csv
    threshold:     float                 # decision threshold (0.67)
    feat_35:       list[str]             # column order for xgb_pipeline.pkl
    model_version: dict = field(default_factory=dict)  # registry version info shown in /health


_state: AppState | None = None


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

def _get_approved_version_metadata() -> dict:
    """
    Query SageMaker Model Registry for the latest Approved version of
    MODEL_GROUP. Returns the CustomerMetadataProperties dict from that version,
    plus 'version_arn' and 'version_number' injected for display in /health.

    Raises RuntimeError if no Approved version exists.
    """
    sm = boto3.client("sagemaker", region_name=REGION)

    paginator = sm.get_paginator("list_model_packages")
    pages = paginator.paginate(
        ModelPackageGroupName=MODEL_GROUP,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending",
    )

    latest = None
    for page in pages:
        packages = page.get("ModelPackageSummaryList", [])
        if packages:
            latest = packages[0]   # most recent Approved version
            break

    if latest is None:
        raise RuntimeError(
            f"No Approved model version found in registry group '{MODEL_GROUP}'. "
            "Approve a version in SageMaker Model Registry before deploying."
        )

    arn     = latest["ModelPackageArn"]
    version = latest.get("ModelPackageVersion", "unknown")

    detail  = sm.describe_model_package(ModelPackageName=arn)
    meta    = detail.get("CustomerMetadataProperties", {})

    meta["version_arn"]    = arn
    meta["version_number"] = str(version)
    meta["approval_status"]= "Approved"

    print(f"  Registry: found Approved version {version} — {arn}")
    return meta


def _s3_download_to_tmp(s3_uri: str, local_name: str) -> Path:
    """
    Download s3://bucket/key to TMP_DIR/local_name.
    Skips the download if the file already exists in /tmp (warm container reuse).
    Returns the local Path.
    """
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    dest = TMP_DIR / local_name

    if dest.exists():
        print(f"    /tmp cache hit — {local_name}")
        return dest

    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got: {s3_uri!r}")

    without_scheme = s3_uri[5:]
    bucket, _, key  = without_scheme.partition("/")

    print(f"    Downloading s3://{bucket}/{key} → {dest}")
    s3 = boto3.client("s3", region_name=REGION)
    s3.download_file(bucket, key, str(dest))
    return dest


def _resolve_model_paths(meta: dict) -> dict[str, Path]:
    """
    Given the registry metadata dict, download every pkl to /tmp and return
    a mapping of logical name → local Path.

    If a metadata key is missing (e.g. older registry version that didn't store
    feat_36 separately), we skip that file — callers handle the None.
    """
    paths = {}
    for logical, meta_key in _META_KEYS.items():
        s3_uri = meta.get(meta_key)
        if not s3_uri:
            print(f"    Warning: metadata key '{meta_key}' not found — skipping {logical}")
            paths[logical] = None
            continue
        local_name = f"{logical}.pkl"
        paths[logical] = _s3_download_to_tmp(s3_uri, local_name)
    return paths


# ---------------------------------------------------------------------------
# Local fallback (non-Lambda dev)
# ---------------------------------------------------------------------------

def _local_model_paths() -> dict[str, Path | None]:
    """Return model paths from the local models/ directory for non-Lambda dev."""
    return {
        "pipeline": MODELS_DIR / "xgb_pipeline.pkl",
        "xgboost":  MODELS_DIR / "xgboost_churn.pkl",
        "encoders": MODELS_DIR / "encoders.pkl",
        "feat_35":  MODELS_DIR / "feature_names_35.pkl",
        "feat_36":  MODELS_DIR / "feature_names_36.pkl",
    }


# ---------------------------------------------------------------------------
# Main initialiser
# ---------------------------------------------------------------------------

def initialize_state() -> "AppState":
    """
    Load and cache all expensive resources. Called once on first request via
    the lazy-init middleware in api/main.py. Subsequent calls return the
    already-cached state immediately.

    On Lambda: queries SageMaker registry → downloads pkls → loads.
    Locally:   loads directly from models/ on disk.
    """
    global _state
    if _state is not None:
        return _state

    is_lambda = bool(os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))
    model_version_meta = {}

    # ------------------------------------------------------------------
    # 1. Resolve model file paths (registry path or local path)
    # ------------------------------------------------------------------
    if is_lambda:
        print("  [Registry] Querying SageMaker Model Registry...")
        registry_meta  = _get_approved_version_metadata()
        model_version_meta = {
            "version_number": registry_meta.get("version_number"),
            "version_arn":    registry_meta.get("version_arn"),
            "approval_status":registry_meta.get("approval_status"),
            "f1_score":       registry_meta.get("f1_score"),
            "roc_auc":        registry_meta.get("roc_auc"),
            "encoding_version": registry_meta.get("encoding_version"),
        }
        print("  [Registry] Resolving S3 paths and downloading to /tmp...")
        paths = _resolve_model_paths(registry_meta)
    else:
        print("  [Local] Non-Lambda environment — loading from models/ directory")
        paths = _local_model_paths()
        model_version_meta = {"source": "local", "version_number": "dev"}

    # ------------------------------------------------------------------
    # 2. Load dataset (same deterministic split as training)
    # ------------------------------------------------------------------
    print("  Loading test set (deterministic split)...")
    raw_df    = load_raw()
    eng_df    = engineer_features(clean(raw_df))
    gbm_df, _ = encode_for_gbm(eng_df)
    _, _, X_test, _, _, y_test = three_way_split(gbm_df)
    eng_test  = eng_df.loc[X_test.index]

    # ------------------------------------------------------------------
    # 3. Load models
    # ------------------------------------------------------------------
    print("  Loading pipeline + feature names...")
    pipeline  = joblib.load(paths["pipeline"])
    feat_35   = joblib.load(paths["feat_35"])

    print("  Loading threshold...")
    if is_lambda:
        # Threshold is stored as a plain string value in registry metadata
        threshold = float(registry_meta["threshold"])
        print(f"    Threshold from registry metadata: {threshold}")
    else:
        threshold_data = joblib.load(MODELS_DIR / "threshold.pkl")
        threshold = float(threshold_data["best_threshold"])

    # ------------------------------------------------------------------
    # 4. Pre-score all test customers
    # ------------------------------------------------------------------
    print("  Pre-scoring all test customers...")
    scores = pipeline.predict_proba(X_test[feat_35])[:, 1]

    # ------------------------------------------------------------------
    # 5. Load SHAP values
    # ------------------------------------------------------------------
    print("  Loading SHAP values...")
    # SHAP values are pre-computed CSV — always loaded from local path
    # (they are static per model version and baked into the image)
    shap_path = (
        TMP_DIR / "shap_values_test.csv"
        if is_lambda and (TMP_DIR / "shap_values_test.csv").exists()
        else MODELS_DIR / "shap_values_test.csv"
    )
    shap_df = pd.read_csv(shap_path, index_col=0)

    # ------------------------------------------------------------------
    # 6. Cache and return
    # ------------------------------------------------------------------
    _state = AppState(
        eng_test_raw=eng_test,
        X_test=X_test,
        y_test=y_test,
        risk_scores=scores,
        shap_df=shap_df,
        threshold=threshold,
        feat_35=feat_35,
        model_version=model_version_meta,
    )

    high_risk = int((scores >= threshold).sum())
    version_label = model_version_meta.get("version_number", "unknown")
    print(
        f"  Ready — {len(eng_test)} customers, "
        f"{high_risk} above threshold ({threshold:.0%}), "
        f"registry version {version_label}"
    )
    return _state


def get_state() -> AppState:
    """FastAPI dependency — inject into route handlers."""
    if _state is None:
        raise RuntimeError("App state not initialised. Is the lifespan running?")
    return _state


def get_customer_raw(customer_id: int, state: AppState) -> dict | None:
    """
    Look up a test customer by pandas index.
    Returns the pre-encoding feature dict, or None if not found.
    """
    if customer_id not in state.eng_test_raw.index:
        return None
    row = state.eng_test_raw.loc[customer_id]
    result = {}
    for col, val in row.items():
        if isinstance(val, (np.integer,)):
            result[col] = int(val)
        elif isinstance(val, (np.floating,)):
            result[col] = float(val)
        else:
            result[col] = val
    return result


def get_top_shap_drivers(customer_id: int, state: AppState, top_n: int = 5) -> list[dict]:
    """
    Pull pre-computed SHAP values for one customer from shap_values_test.csv.
    Returns top_n drivers sorted by |SHAP| descending.
    """
    if customer_id not in state.shap_df.index:
        return []

    row   = state.shap_df.loc[customer_id]
    items = [(feat, float(val)) for feat, val in row.items()]
    items.sort(key=lambda x: abs(x[1]), reverse=True)

    return [
        {
            "feature":    feat,
            "shap_value": round(sv, 4),
            "raw_value":  0.0,
            "direction":  "increases churn risk" if sv > 0 else "decreases churn risk",
        }
        for feat, sv in items[:top_n]
    ]