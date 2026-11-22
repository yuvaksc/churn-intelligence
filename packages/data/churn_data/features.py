"""churn_data.features — feature engineering as a fitted sklearn transformer.

`engineer_features()` in loader.py computes two statistics from whatever frame it
is given: the median monthly charge (High Risk Flag) and the per-bundle median
charge (Charge Index). Applied to a 50-row upload, those medians differ from
training and the features silently drift.

`FeatureEngineer` learns those medians once in `fit` (training data) and reuses
them in `transform`, so it can sit inside the model's sklearn Pipeline and be
saved with it: training and serving compute identical features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

SERVICE_COLS = [
    "Online Security", "Online Backup", "Device Protection",
    "Tech Support", "Streaming TV", "Streaming Movies",
]
ENGINEERED = [
    "Charge Ratio", "Avg Monthly Charge", "Services Count", "High Risk Flag",
    "Tenure Bin", "Sticky Payment", "Charge Index",
]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds the 7 engineered features using statistics learned at fit time."""

    def fit(self, X: pd.DataFrame, y=None):
        X = pd.DataFrame(X)
        self.median_charge_ = float(X["Monthly Charges"].median())
        if {"Internet Service", "Phone Service"} <= set(X.columns):
            med = X.groupby(["Internet Service", "Phone Service"])["Monthly Charges"].median()
            self.bundle_medians_ = {f"{k[0]}|{k[1]}": float(v) for k, v in med.items()}
        else:
            self.bundle_medians_ = {}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(X).copy()
        total = pd.to_numeric(df["Total Charges"], errors="coerce").fillna(0.0)
        df["Charge Ratio"] = df["Monthly Charges"] / (total + 1e-9)
        df["Avg Monthly Charge"] = total / (df["Tenure Months"] + 1e-9)
        df["Services Count"] = sum(
            (df[c] == "Yes").astype(int) for c in SERVICE_COLS if c in df.columns
        ) if any(c in df.columns for c in SERVICE_COLS) else 0
        df["High Risk Flag"] = (
            (df["Contract"] == "Month-to-month") & (df["Monthly Charges"] > self.median_charge_)
        ).astype(int)
        df["Tenure Bin"] = pd.cut(
            df["Tenure Months"], bins=[0, 12, 36, 60, float("inf")],
            labels=[0, 1, 2, 3], include_lowest=True,
        ).astype(int)
        df["Sticky Payment"] = (
            (df.get("Paperless Billing", "No") == "Yes")
            & df.get("Payment Method", pd.Series("", index=df.index))
                .astype(str).str.contains("automatic", case=False, na=False)
        ).astype(int)
        if self.bundle_medians_ and {"Internet Service", "Phone Service"} <= set(df.columns):
            keys = df["Internet Service"].astype(str) + "|" + df["Phone Service"].astype(str)
            medians = keys.map(self.bundle_medians_).fillna(self.median_charge_)
            df["Charge Index"] = df["Monthly Charges"] / (medians + 1e-9)
        else:
            df["Charge Index"] = 1.0
        return df

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else []
        return np.asarray(base + [c for c in ENGINEERED if c not in base], dtype=object)
