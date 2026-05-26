"""
run_agents.py — War Room entrypoint.

Usage:
    # Single customer from test set (by row index, 0-based)
    python run_agents.py --index 0

    # Single customer — pick the first HIGH-RISK customer automatically
    python run_agents.py --auto

    # Batch: run on first N high-risk test customers
    python run_agents.py --batch --top_n 5

    # Specify US state for competitor intel
    python run_agents.py --index 0 --state California

Outputs:
    Console:  full war room report per customer
    File:     models/reports/retention_log.jsonl   (appended per run)
    File:     models/reports/war_room_batch.json   (batch mode only)
"""

import sys
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import asyncio

sys.path.insert(0, "src")

from data.loader import (
    load_raw, clean, engineer_features,
    encode_for_gbm, three_way_split,
)
from agents.graph import war_room_graph


# ── Test set reconstruction ───────────────────────────────────────────────────

def load_test_set() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Reconstructs the test split using the same random_state=42 as training.
    Returns:
        eng_test_raw  — pre-encoding test rows (human-readable values for agents)
        X_test_enc    — encoded test rows (for risk score pre-screening)
        y_test        — true labels
    """
    raw_df   = load_raw()
    clean_df = clean(raw_df)
    eng_df   = engineer_features(clean_df)
    gbm_df, _ = encode_for_gbm(eng_df)
    X_train, X_val, X_test, y_train, y_val, y_test = three_way_split(gbm_df)

    # Pre-encoding rows for the test indices
    eng_test_raw = eng_df.loc[X_test.index]
    return eng_test_raw, X_test, y_test


def row_to_dict(row: pd.Series) -> dict:
    """Convert a pre-encoding DataFrame row to a plain dict."""
    result = {}
    for col, val in row.items():
        if isinstance(val, (np.integer,)):
            result[col] = int(val)
        elif isinstance(val, (np.floating,)):
            result[col] = float(val)
        else:
            result[col] = val
    return result


# def get_high_risk_indices(X_test: pd.DataFrame, top_n: int = 20) -> list[int]:
#     """
#     Use the pipeline to rank test customers by risk score.
#     Returns positional indices (iloc) into eng_test_raw for the top_n highest risk.
#     """
#     pipeline  = joblib.load("models/xgb_pipeline.pkl")
#     feat_35   = joblib.load("models/feature_names_35.pkl")
#     threshold = joblib.load("models/threshold.pkl")["best_threshold"]

#     scores    = pipeline.predict_proba(X_test[feat_35])[:, 1]
#     high_mask = scores >= threshold
#     high_idxs = np.where(high_mask)[0]                     # positional
#     # sorted_i  = high_idxs[np.argsort(-scores[high_idxs])]  # sort by score desc
#     # return sorted_i[:top_n].tolist()

#     from collections import defaultdict
#     bins = defaultdict(list)
#     for i in high_idxs:
#         tenure = int(eng_test_raw.iloc[i].get("Tenure Bin", 0))
#         bins[tenure].append((scores[i], i))
#     diverse = []
#     for b in sorted(bins):
#         best = sorted(bins[b], reverse=True)[0][1]
#         diverse.append(best)
#     return diverse[:top_n]
# run_agents.py — replace get_high_risk_indices() entirely

def get_high_risk_indices(X_test: pd.DataFrame, top_n: int = 20) -> list[int]:
    from collections import defaultdict

    pipeline  = joblib.load("models/xgb_pipeline.pkl")
    feat_35   = joblib.load("models/feature_names_35.pkl")
    threshold = joblib.load("models/threshold.pkl")["best_threshold"]

    scores    = pipeline.predict_proba(X_test[feat_35])[:, 1]
    high_mask = scores >= threshold
    high_idxs = np.where(high_mask)[0]

    if len(high_idxs) == 0:
        return []

    # Bin by Tenure Bin (already encoded in X_test as 0/1/2/3)
    # Pick the highest-scoring customer from each bin → diverse archetypes
    bins: dict = defaultdict(list)
    for i in high_idxs:
        tenure_bin = int(X_test.iloc[i]["Tenure Bin"])
        bins[tenure_bin].append((float(scores[i]), i))

    diverse = []
    for b in sorted(bins):                              # bins 0,1,2,3 in order
        best = sorted(bins[b], reverse=True)[0][1]     # highest score in that bin
        diverse.append(best)

    return diverse[:top_n]

# ── War Room runner ───────────────────────────────────────────────────────────

def run_war_room(customer_id: str, customer_raw: dict, customer_state: str = "DEFAULT") -> dict:
    initial_state = {
        "customer_id":     customer_id,
        "customer_raw":    customer_raw,
        "customer_state":  customer_state,
        # Agent 1 outputs (defaults)
        "risk_score":      0.0,
        "risk_label":      "PENDING",
        "shap_drivers":    [],
        "risk_summary":    "",
        # Agent 2 outputs (defaults)
        "similar_profiles": [],
        "churn_reasons":   [],
        "evidence_report": "",
        # Agent 3 outputs (defaults)
        "policy":          {},
        "competitor_intel": {},
        "retention_offer": "",
        "crm_log_id":      "",
        "crm_logged":      False,
        # LangGraph internals
        "messages":        [],
    }

    print(f"\n{'═' * 60}")
    print(f"  WAR ROOM  ›  Customer {customer_id}")
    print(f"{'═' * 60}")

    return asyncio.run(war_room_graph.ainvoke(initial_state))


# ── Report printer ────────────────────────────────────────────────────────────

def print_report(result: dict) -> None:
    sep = "─" * 60
    print(f"\n{sep}")
    print("  FINAL WAR ROOM REPORT")
    print(sep)
    print(f"  Customer:     {result['customer_id']}")
    print(f"  Risk Score:   {result['risk_score']:.1%}  [{result['risk_label']}]")

    if result["risk_label"] != "HIGH":
        print(f"  Decision:     No action required (below intervention threshold)")
        print(sep)
        return

    print(f"\n  ── RISK SUMMARY ───────────────────────────────────")
    print(f"  {result['risk_summary']}")

    print(f"\n  ── EVIDENCE ───────────────────────────────────────")
    print(f"  {result['evidence_report']}")

    print(f"\n  ── RETENTION OFFER ────────────────────────────────")
    for line in result["retention_offer"].split("\n"):
        print(f"  {line}")

    print(f"\n  ── CRM LOG ────────────────────────────────────────")
    print(f"  Log ID:       {result['crm_log_id']}")
    print(f"  Status:       {'✓ Logged' if result['crm_logged'] else '✗ Not logged'}")
    print(f"  File:         models/reports/retention_log.jsonl")
    print(sep)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Churn Intelligence War Room")
    parser.add_argument("--index", type=int,  default=None,
                        help="Test set positional index (0-based)")
    parser.add_argument("--auto",  action="store_true",
                        help="Auto-select the highest-risk test customer")
    parser.add_argument("--batch", action="store_true",
                        help="Run on top_n high-risk customers")
    parser.add_argument("--top_n", type=int,  default=3,
                        help="Number of customers to run in batch mode (default: 3)")
    parser.add_argument("--state", type=str,  default="DEFAULT",
                        help="US state for competitor lookup (e.g. 'California')")
    args = parser.parse_args()

    print("Loading test set (deterministic split, random_state=42)...")
    eng_test_raw, X_test, y_test = load_test_set()

    # ── Single: auto ──────────────────────────────────────────────────────────
    if args.auto or (args.index is None and not args.batch):
        high_risk = get_high_risk_indices(X_test, top_n=1)
        if not high_risk:
            print("No customers above threshold in test set.")
            return
        pos = high_risk[0]
        row  = eng_test_raw.iloc[pos]
        cid  = f"TEST-{eng_test_raw.index[pos]}"
        result = run_war_room(cid, row_to_dict(row), args.state)
        print_report(result)

    # ── Single: by index ──────────────────────────────────────────────────────
    elif args.index is not None:
        row = eng_test_raw.iloc[args.index]
        cid = f"TEST-{eng_test_raw.index[args.index]}"
        result = run_war_room(cid, row_to_dict(row), args.state)
        print_report(result)

    # ── Batch ─────────────────────────────────────────────────────────────────
    elif args.batch:
        high_risk = get_high_risk_indices(X_test, top_n=args.top_n)
        print(f"\nBatch mode: {len(high_risk)} high-risk customers")

        all_results = []
        for pos in high_risk:
            row    = eng_test_raw.iloc[pos]
            cid    = f"TEST-{eng_test_raw.index[pos]}"
            result = run_war_room(cid, row_to_dict(row), args.state)
            print_report(result)
            all_results.append({
                k: v for k, v in result.items() if k != "messages"
            })

        out_path = Path("models/reports/war_room_batch.json")
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nBatch report saved → {out_path}")


if __name__ == "__main__":
    main()