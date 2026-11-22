"""churn_ml.model — the single sklearn Pipeline used for training AND serving.

    raw cleaned columns (as stored in customers.all_features)
      └─ FeatureEngineer      learns medians at fit time (churn_data.features)
      └─ ColumnTransformer    ordinal Contract · one-hot binary/nominal · numeric passthrough
      └─ classifier           LogisticRegression (baseline) | XGBClassifier

The whole object is saved, so the API never re-implements encoding: whatever
was fitted is exactly what scores.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from churn_data.features import ENGINEERED, FeatureEngineer

CONTRACT = ["Contract"]
CATEGORICAL = [
    "Gender", "Senior Citizen", "Partner", "Dependents", "Phone Service", "Paperless Billing",
    "Multiple Lines", "Internet Service", "Online Security", "Online Backup", "Device Protection",
    "Tech Support", "Streaming TV", "Streaming Movies", "Payment Method",
]
NUMERIC = ["Tenure Months", "Monthly Charges", "Total Charges"] + ENGINEERED
RAW_FEATURES = CONTRACT + CATEGORICAL + ["Tenure Months", "Monthly Charges", "Total Charges"]
TARGET = "Churn Value"


def preprocessor() -> Pipeline:
    encode = ColumnTransformer(
        [
            ("contract", OrdinalEncoder(categories=[["Month-to-month", "One year", "Two year"]],
                                        handle_unknown="use_encoded_value", unknown_value=-1), CONTRACT),
            ("categorical", OneHotEncoder(drop="if_binary", handle_unknown="ignore",
                                          sparse_output=False), CATEGORICAL),
            ("numeric", "passthrough", NUMERIC),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([("features", FeatureEngineer()), ("encode", encode)])


def logistic_baseline() -> Pipeline:
    return Pipeline([
        ("prep", preprocessor()),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
    ])


def xgboost_pipeline(scale_pos_weight: float, **params) -> Pipeline:
    from xgboost import XGBClassifier
    return Pipeline([
        ("prep", preprocessor()),
        ("clf", XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                              scale_pos_weight=scale_pos_weight, random_state=42,
                              n_jobs=-1, **params)),
    ])


def to_frame(rows: list[dict]) -> pd.DataFrame:
    """Raw feature dicts (customers.all_features) → model input frame."""
    df = pd.DataFrame(rows)
    # Sparse records (e.g. from an integration) still score: absent categoricals become
    # unknown categories (ignored by the one-hot encoder), absent numerics become 0.
    for col in RAW_FEATURES:
        if col not in df.columns:
            df[col] = None
    for col in ["Tenure Months", "Monthly Charges", "Total Charges"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df
