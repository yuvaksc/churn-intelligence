"""churn_ml.scoring — score every customer with the champion and explain it with SHAP.

    python -m churn_ml.scoring --seed   # first time: load the customers, then score them
    python -m churn_ml.scoring          # after a new champion: re-score everyone

For every customer: the calibrated churn probability, HIGH/LOW at the economic threshold,
and the top-5 SHAP drivers (TreeExplainer on the fitted XGBoost). Results go to the
`predictions` table and onto each customer row, so the API and Agent 1 just read them:
SHAP is computed here, once per batch, never while a user waits.

--seed first loads the dashboard's customers: the held-out TEST split (same seed as
training, so the model never saw them), plus the synthetic CRM tickets the MCP tools read.
Idempotent: re-running updates rows in place. Needs DATABASE_URL and a champion model.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from churn_ml.model import TARGET, to_frame

REPORTS = Path("models/reports")
TOP_N = 5
SEED_VERSION = "v3"
SERVICE_COLS = ["Online Security", "Online Backup", "Device Protection",
                "Tech Support", "Streaming TV", "Streaming Movies"]


# ── SHAP ─────────────────────────────────────────────────────────────────────

def _fitted_pipeline(model):
    """The prefit pipeline inside CalibratedClassifierCV (or the model itself)."""
    if hasattr(model, "calibrated_classifiers_"):
        return model.calibrated_classifiers_[0].estimator
    return model


def explain(model, frame: pd.DataFrame, top_n: int = TOP_N) -> list[list[dict]]:
    """Top-N SHAP drivers per row. SHAP explains the XGBoost margin; isotonic
    calibration is monotonic, so the driver ranking is unchanged."""
    import shap

    pipe = _fitted_pipeline(model)
    prep, clf = pipe.named_steps["prep"], pipe.named_steps["clf"]
    X = prep.transform(frame)
    names = list(prep.named_steps["encode"].get_feature_names_out())
    values = np.asarray(shap.TreeExplainer(clf).shap_values(X))
    out = []
    for row_vals, row_x in zip(values, np.asarray(X, dtype=float)):
        order = np.argsort(-np.abs(row_vals))[:top_n]
        out.append([{
            "feature": names[i],
            "shap_value": round(float(row_vals[i]), 4),
            "raw_value": round(float(row_x[i]), 4),
            "direction": "increases churn risk" if row_vals[i] > 0 else "decreases churn risk",
        } for i in order])
    return out


# ── Scoring ──────────────────────────────────────────────────────────────────

def score(customers: list[dict], predictor) -> list[dict]:
    """customers: [{customer_id, all_features}] → prediction dicts with SHAP drivers."""
    frame = to_frame([c["all_features"] for c in customers])
    probs = predictor.model.predict_proba(frame)[:, 1]
    drivers = explain(predictor.model, frame)
    return [{
        "customer_id": c["customer_id"], "model_version": predictor.model_version,
        "risk_score": round(float(p), 4), "risk_label": "HIGH" if p >= predictor.threshold else "LOW",
        "threshold": predictor.threshold, "shap_drivers": d,
    } for c, p, d in zip(customers, probs, drivers)]


def run(predictor) -> dict:
    """Score every customer in the database and store the results."""
    from churn_data import db

    with db.tx() as c:
        customers = c.execute("select customer_id, all_features from customers").fetchall()
    preds = score(customers, predictor) if customers else []
    db.insert_predictions(preds)
    db.set_meta("threshold", str(predictor.threshold))
    db.set_meta("model_version", predictor.model_version)

    summary = {"scored": len(preds), "model_version": predictor.model_version,
               "high_risk": sum(p["risk_label"] == "HIGH" for p in preds),
               "scored_at": datetime.now(timezone.utc).isoformat()}
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "scoring_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


# ── Seeding the dashboard's customers ────────────────────────────────────────

def _jsonable(row: pd.Series) -> dict:
    out = {}
    for col, val in row.items():
        if col == TARGET:                     # never hand the label to the agents
            continue
        if isinstance(val, np.integer):
            val = int(val)
        elif isinstance(val, np.floating):
            val = float(val)
        out[col] = val
    return out


def test_customers() -> list[dict]:
    from churn_data.loader import clean, engineer_features, load_raw, three_way_split

    eng = engineer_features(clean(load_raw()))
    _, _, X_test, _, _, y_test = three_way_split(eng)
    rows = []
    for idx in X_test.index:
        r = eng.loc[idx]
        rows.append({
            "customer_id": int(idx),
            "contract": str(r["Contract"]), "monthly_charges": float(r["Monthly Charges"]),
            "tenure_months": int(r["Tenure Months"]), "internet_service": str(r["Internet Service"]),
            "payment_method": str(r["Payment Method"]),
            "services_count": int(sum(r[c] == "Yes" for c in SERVICE_COLS)),
            "high_risk_flag": int(r["High Risk Flag"]),
            "true_label": int(y_test.loc[idx]),
            "all_features": _jsonable(r),
        })
    return rows


def seed() -> dict:
    from churn_data import db
    from churn_data.crm import tickets_for

    customers = test_customers()
    db.upsert_customers(customers)
    tickets = [t for c in customers for t in tickets_for(c["customer_id"], c["all_features"])]
    db.replace_crm_tickets(tickets)
    db.set_meta("seed_version", SEED_VERSION)
    return {"customers": len(customers), "crm_tickets": len(tickets)}


def main(argv=None) -> None:
    from churn_ml.predictor import get_predictor

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", action="store_true", help="load the test-split customers + CRM tickets first")
    args = ap.parse_args(argv)
    out = seed() if args.seed else {}
    out.update(run(get_predictor()))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
