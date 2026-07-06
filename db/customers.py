"""db/customers.py — customers table: seed + read helpers (raw SQL, sync).

seed_customers() reproduces the exact computation that api/dependencies.py
initialize_state() used to do in memory (deterministic test split + pre-scoring
+ SHAP lookup) and writes one row per test customer. The dashboard then serves
list/detail straight from SQL; the war room still scores fresh via agent1.

All functions are blocking — async callers wrap them in asyncio.to_thread.
"""

import sys
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from db.connection import connect

# Reuse the existing preprocessing pipeline (src is added to the path the same
# way api/dependencies.py does it).
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data.loader import (  # noqa: E402
    load_raw, clean, engineer_features,
    encode_for_gbm, three_way_split,
)

MODELS_DIR   = Path("models")
SEED_VERSION = "1"

_SUMMARY_COLS = (
    "customer_id", "risk_score", "risk_label", "contract", "monthly_charges",
    "tenure_months", "internet_service", "services_count", "high_risk_flag",
    "true_label",
)


# ── Seeding ──────────────────────────────────────────────────────────────────

def _row_to_dict(row: pd.Series) -> dict:
    """Coerce a pre-encoding feature row to JSON-safe Python types (mirrors the
    old get_customer_raw)."""
    result: dict = {}
    for col, val in row.items():
        if isinstance(val, np.integer):
            result[col] = int(val)
        elif isinstance(val, np.floating):
            result[col] = float(val)
        else:
            result[col] = val
    return result


def _top_shap(shap_df: pd.DataFrame, idx, top_n: int = 5) -> list[dict]:
    """Top-N SHAP drivers for one customer (mirrors the old get_top_shap_drivers)."""
    if idx not in shap_df.index:
        return []
    row   = shap_df.loc[idx]
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


def _compute_customer_rows() -> tuple[list[tuple], float]:
    """Run the deterministic pipeline + pre-scoring; return insert tuples and the
    threshold. Mirrors api/dependencies.initialize_state()."""
    raw_df    = load_raw()
    eng_df    = engineer_features(clean(raw_df))
    gbm_df, _ = encode_for_gbm(eng_df)
    _, _, X_test, _, _, y_test = three_way_split(gbm_df)
    eng_test  = eng_df.loc[X_test.index]

    pipeline  = joblib.load(MODELS_DIR / "xgb_pipeline.pkl")
    feat_35   = joblib.load(MODELS_DIR / "feature_names_35.pkl")
    threshold = float(joblib.load(MODELS_DIR / "threshold.pkl")["best_threshold"])
    scores    = pipeline.predict_proba(X_test[feat_35])[:, 1]
    shap_df   = pd.read_csv(MODELS_DIR / "shap_values_test.csv", index_col=0)

    rows: list[tuple] = []
    for pos, idx in enumerate(eng_test.index):
        score = float(scores[pos])
        row   = eng_test.loc[idx]
        rows.append((
            int(idx),
            round(score, 4),
            "HIGH" if score >= threshold else "LOW",
            str(row.get("Contract", "")),
            float(row.get("Monthly Charges", 0.0)),
            int(row.get("Tenure Months", 0)),
            str(row.get("Internet Service", "")),
            int(row.get("Services Count", 0)),
            int(row.get("High Risk Flag", 0)),
            int(y_test.loc[idx]) if idx in y_test.index else None,
            json.dumps(_row_to_dict(row)),
            json.dumps(_top_shap(shap_df, idx)),
        ))
    return rows, threshold


def seed_customers(force: bool = False) -> int:
    """Populate the customers table if empty (or force-rebuild). Returns the row
    count. Blocking — call via asyncio.to_thread."""
    with connect() as conn:
        if force:
            conn.execute("DELETE FROM customers")
            conn.commit()
        else:
            existing = conn.execute("SELECT COUNT(*) AS n FROM customers").fetchone()["n"]
            if existing > 0:
                return existing

        rows, threshold = _compute_customer_rows()

        conn.executemany(
            """INSERT OR REPLACE INTO customers
               (customer_id, risk_score, risk_label, contract, monthly_charges,
                tenure_months, internet_service, services_count, high_risk_flag,
                true_label, all_features, top_shap_drivers)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        conn.executemany(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            [("threshold", str(threshold)), ("seed_version", SEED_VERSION)],
        )
        conn.commit()
        return len(rows)


# ── Reads ────────────────────────────────────────────────────────────────────

def _summary(row) -> dict:
    return {k: row[k] for k in _SUMMARY_COLS}


def list_customers(limit: int, offset: int, risk_only: bool) -> list[dict]:
    sql    = "SELECT * FROM customers"
    params: list = []
    if risk_only:
        sql += " WHERE risk_label = 'HIGH'"
    sql += " ORDER BY risk_score DESC LIMIT ? OFFSET ?"
    params += [limit, offset]
    with connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_summary(r) for r in rows]


def get_customer(customer_id: int) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM customers WHERE customer_id = ?", (customer_id,)).fetchone()
    if row is None:
        return None
    detail = _summary(row)
    detail["all_features"]     = json.loads(row["all_features"])
    detail["top_shap_drivers"] = json.loads(row["top_shap_drivers"])
    return detail


def get_customer_features(customer_id: int) -> dict | None:
    """The full pre-encoding feature dict the war room feeds to the agents."""
    with connect() as conn:
        row = conn.execute("SELECT all_features FROM customers WHERE customer_id = ?", (customer_id,)).fetchone()
    return json.loads(row["all_features"]) if row else None


def fetch_scoring_rows() -> list[dict]:
    """risk_score / true_label / monthly_charges for every customer — used by the
    metrics fallback."""
    with connect() as conn:
        rows = conn.execute("SELECT risk_score, true_label, monthly_charges FROM customers").fetchall()
    return [dict(r) for r in rows]


def count_customers() -> int:
    with connect() as conn:
        return conn.execute("SELECT COUNT(*) AS n FROM customers").fetchone()["n"]


def count_high_risk() -> int:
    with connect() as conn:
        return conn.execute("SELECT COUNT(*) AS n FROM customers WHERE risk_label = 'HIGH'").fetchone()["n"]


def get_threshold(default: float = 0.5) -> float:
    with connect() as conn:
        row = conn.execute("SELECT value FROM meta WHERE key = 'threshold'").fetchone()
    return float(row["value"]) if row else default
