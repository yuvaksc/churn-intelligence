"""churn_ml.economics — the business assumptions behind the decision threshold.

One source of truth for the retention economics, shared by threshold selection
(training) and the /api/metrics ROI figures.

Intervening on a customer costs an offer; it pays off only if they would have
churned AND the offer saves them. With a *calibrated* probability p:

    expected value of contacting = p · SUCCESS · monthly · LTV_MONTHS − offer_cost(monthly)

so the break-even probability for one customer is
    p* = offer_cost(monthly) / (SUCCESS · monthly · LTV_MONTHS)
That reasoning is only valid when p is a real probability, which is why the
model is calibrated before the threshold is chosen.
"""

from __future__ import annotations

import numpy as np

RETENTION_SUCCESS = 0.30   # share of contacted churners we actually save
LTV_MONTHS = 24            # months of revenue retained per saved customer
DISCOUNT_CREDIT = 20.0     # one-time credit per offer ($)
DISCOUNT_PCT = 0.10        # recurring discount on the monthly bill
DISCOUNT_MONTHS = 6        # how long the discount runs

ASSUMPTIONS = {
    "retention_success_rate": RETENTION_SUCCESS,
    "ltv_months": LTV_MONTHS,
    "discount_credit_usd": DISCOUNT_CREDIT,
    "discount_pct": DISCOUNT_PCT,
    "discount_months": DISCOUNT_MONTHS,
}


def offer_cost(monthly):
    return DISCOUNT_CREDIT + np.asarray(monthly) * DISCOUNT_PCT * DISCOUNT_MONTHS


def break_even_probability(monthly):
    monthly = np.asarray(monthly, dtype=float)
    return offer_cost(monthly) / (RETENTION_SUCCESS * monthly * LTV_MONTHS)


def net_value(y_true, flagged, monthly) -> float:
    """Realised value of a campaign: saved LTV among true churners we flagged,
    minus the cost of every offer sent."""
    y_true, flagged, monthly = map(np.asarray, (y_true, flagged, monthly))
    saved = (flagged & (y_true == 1)) * RETENTION_SUCCESS * monthly * LTV_MONTHS
    cost = flagged * offer_cost(monthly)
    return float(saved.sum() - cost.sum())


def expected_net_value(p, flagged, monthly) -> float:
    """Same, in expectation under calibrated probabilities (no labels needed)."""
    p, flagged, monthly = map(np.asarray, (p, flagged, monthly))
    return float((flagged * (p * RETENTION_SUCCESS * monthly * LTV_MONTHS - offer_cost(monthly))).sum())


def choose_threshold(y_true, p, monthly, grid=None) -> tuple[float, float]:
    """Global threshold maximising realised net value on held-out data."""
    grid = np.round(np.arange(0.05, 0.95, 0.01), 2) if grid is None else grid
    values = [net_value(y_true, np.asarray(p) >= t, monthly) for t in grid]
    best = int(np.argmax(values))
    return float(grid[best]), float(values[best])
