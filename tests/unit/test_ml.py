"""ML units: threshold economics and the serving pipeline contract.
Uses the logistic-regression pipeline (sklearn only) so it runs without xgboost."""

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV

from churn_data.loader import clean, load_raw
from churn_ml import economics
from churn_ml.model import RAW_FEATURES, TARGET, logistic_baseline
from churn_ml.predictor import PipelinePredictor


# ── economics / threshold math ───────────────────────────────────────────────
def test_break_even_probability_formula():
    m = 70.0
    expected = (economics.DISCOUNT_CREDIT + m * economics.DISCOUNT_PCT * economics.DISCOUNT_MONTHS) / (
        economics.RETENTION_SUCCESS * m * economics.LTV_MONTHS)
    assert economics.break_even_probability(m) == pytest.approx(expected)


def test_expected_value_is_zero_at_break_even():
    m = np.array([70.0])
    p = economics.break_even_probability(m)
    assert economics.expected_net_value(p, np.array([True]), m) == pytest.approx(0.0, abs=1e-9)


def test_net_value_counts_saved_churners_and_every_offer():
    y = np.array([1, 0, 1])
    flagged = np.array([True, True, False])
    m = np.array([100.0, 100.0, 100.0])
    saved = economics.RETENTION_SUCCESS * 100 * economics.LTV_MONTHS
    assert economics.net_value(y, flagged, m) == pytest.approx(saved - 2 * economics.offer_cost(100.0))


def test_choose_threshold_prefers_flagging_the_churners():
    y = np.array([1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.3, 0.2, 0.1, 0.05])
    m = np.full(6, 80.0)
    t, value = economics.choose_threshold(y, p, m)
    assert 0.3 < t <= 0.8
    assert value == pytest.approx(economics.net_value(y, p >= t, m))


# ── serving: one pipeline, raw rows in, calibrated probability out ──────────
@pytest.fixture(scope="module")
def predictor():
    df = clean(load_raw())
    X, y = df[RAW_FEATURES], df[TARGET]
    fitted = logistic_baseline().fit(X.iloc[:4000], y.iloc[:4000])
    model = CalibratedClassifierCV(fitted, method="isotonic", cv="prefit").fit(X.iloc[4000:5500], y.iloc[4000:5500])
    return PipelinePredictor(model, threshold=0.3, version="test:1"), df


def test_predictor_scores_raw_feature_dicts(predictor):
    pred, df = predictor
    raw = df[RAW_FEATURES].iloc[6000].to_dict()
    out = pred.predict(raw)
    assert 0.0 <= out["risk_score"] <= 1.0
    assert out["risk_label"] == ("HIGH" if out["risk_score"] >= 0.3 else "LOW")
    assert out["shap_drivers"] == []          # SHAP comes from the batch job


def test_predictor_tolerates_unseen_categories_and_string_numbers(predictor):
    pred, df = predictor
    raw = df[RAW_FEATURES].iloc[6001].to_dict()
    raw.update({"Payment Method": "Crypto", "Monthly Charges": "70.5", "Total Charges": ""})
    assert 0.0 <= pred.predict(raw)["risk_score"] <= 1.0


def test_batch_and_single_scoring_agree(predictor):
    pred, df = predictor
    rows = df[RAW_FEATURES].iloc[6000:6010].to_dict(orient="records")
    batch = pred.predict_proba(rows)
    single = [pred.predict(r)["risk_score"] for r in rows]
    assert np.allclose(np.round(batch, 4), single)


def test_predictor_scores_sparse_records(predictor):
    pred, _ = predictor
    out = pred.predict({"Contract": "Month-to-month", "Monthly Charges": 90, "Tenure Months": 2,
                        "Internet Service": "Fiber optic"})
    assert 0.0 <= out["risk_score"] <= 1.0
