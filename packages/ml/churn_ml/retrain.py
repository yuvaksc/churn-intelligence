"""churn_ml.retrain — train → register challenger → compare → promote.

    python -m churn_ml.retrain                    # promote only if the challenger wins
    python -m churn_ml.retrain --promote never    # evaluate + register, keep the champion
    python -m churn_ml.retrain --promote force    # first deploy / manual override
    python -m churn_ml.retrain --n-iter 10        # smaller search (CI smoke run)

Writes models/reports/retrain_report.json (also uploaded by the retrain workflow)
and, on promotion, models/champion.joblib for the API.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from churn_ml import registry
from churn_ml.training import train

REPORT = Path("models/reports/retrain_report.json")


def run(n_iter: int = 25, promote: str = "auto") -> dict:
    result = train(n_iter=n_iter)
    logged = registry.log_candidate(result)
    comparison = registry.compare(result)

    should_promote = promote == "force" or (promote == "auto" and comparison["promote"])
    if should_promote:
        registry.promote(logged["version"], result)

    report = {
        "challenger": {"version": logged["version"], "run_id": logged["run_id"], **result.metrics},
        "baseline_lr": result.baseline_metrics,
        "lift_vs_baseline": {k: round(result.metrics[k] - result.baseline_metrics[k], 4)
                             for k in ("roc_auc", "pr_auc", "brier", "recall", "net_value_usd")},
        "champion_before": comparison["champion"],
        "decision": {"promoted": should_promote, "mode": promote, "reason": comparison["reason"]},
        "params": result.params,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-iter", type=int, default=25, help="random-search candidates (5-fold CV each)")
    ap.add_argument("--promote", choices=["auto", "never", "force"], default="auto")
    args = ap.parse_args(argv)
    report = run(args.n_iter, args.promote)
    print(json.dumps({"challenger": report["challenger"], "decision": report["decision"],
                      "lift_vs_baseline": report["lift_vs_baseline"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
