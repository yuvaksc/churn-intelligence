"""
Data loading and preprocessing for IBM Telco Customer Churn dataset.

Encoding strategy (this matters significantly for F1):
  - Binary Yes/No columns  -> LabelEncoder (0/1)
  - Nominal 3+ categories  -> OneHotEncoder (no false ordinal relationship)
  - CatBoost path          -> raw strings returned separately, CatBoost
                             handles categoricals natively via ordered target
                             statistics -- better than any manual encoding
  - Numeric columns        -> passthrough

Three-way split: 60% train / 20% val / 20% test
  - val  -> threshold tuning (not the test set -- that was mild leakage before)
  - test -> evaluated exactly once at the very end
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

RAW_DIR    = Path("data/raw")
CSV_PATH   = RAW_DIR / "telco.csv"
TARGET_COL = "Churn Value"

GEO_COLS = [
    "CustomerID", "Count", "Country", "State", "City",
    "Zip Code", "Lat Long", "Latitude", "Longitude",
]
LEAKAGE_COLS = [
    "Churn Label",
    "Churn Score",
    "CLTV",
]
RAG_TEXT_COL = "Churn Reason"

BINARY_COLS = [
    "Gender", "Senior Citizen", "Partner", "Dependents",
    "Phone Service", "Paperless Billing",
]
NOMINAL_COLS = [
    "Multiple Lines", "Internet Service",
    "Online Security", "Online Backup", "Device Protection",
    "Tech Support", "Streaming TV", "Streaming Movies",
    "Payment Method",                    # Contract removed — gets ordinal encoding
]

# Contract has a genuine ordinal hierarchy: commitment length in months
CONTRACT_ORDINAL = {"Month-to-month": 1, "One year": 12, "Two year": 24}


def load_raw() -> pd.DataFrame:
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {CSV_PATH}\n"
            f"Place your telco.csv file in the data/raw/ directory."
        )
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()
    print(f"Raw dataset: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def extract_rag_corpus(df: pd.DataFrame) -> None:
    # Skip in Lambda — read-only filesystem and not needed at serving time
    import os
    if os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
        return
    rag_cols = [c for c in ["CustomerID", "Churn Value", RAG_TEXT_COL] if c in df.columns]
    rag_df   = df[rag_cols].dropna(subset=[RAG_TEXT_COL])
    rag_df   = rag_df[rag_df[RAG_TEXT_COL].str.strip() != ""]
    out_path = RAW_DIR / "churn_reasons_rag.csv"
    rag_df.to_csv(out_path, index=False)
    print(f"RAG corpus: {len(rag_df)} churn reasons -> {out_path}")

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    extract_rag_corpus(df)
    cols_to_drop = [c for c in GEO_COLS + LEAKAGE_COLS + [RAG_TEXT_COL] if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"Dropped {len(cols_to_drop)} non-feature columns")
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce").fillna(0.0)
    assert df[TARGET_COL].isin([0, 1]).all(), "Unexpected values in Churn Value"
    print(f"Churn rate: {df[TARGET_COL].mean():.1%}  ({df[TARGET_COL].sum()} / {len(df)})")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Charge Ratio"] = df["Monthly Charges"] / (df["Total Charges"] + 1e-9)
    df["Avg Monthly Charge"] = df["Total Charges"] / (df["Tenure Months"] + 1e-9)

    service_cols = [
        "Online Security", "Online Backup", "Device Protection",
        "Tech Support", "Streaming TV", "Streaming Movies",
    ]
    df["Services Count"] = sum(
        (df[c] == "Yes").astype(int) for c in service_cols if c in df.columns
    )

    median_charge = df["Monthly Charges"].median()
    df["High Risk Flag"] = (
        (df["Contract"] == "Month-to-month") &
        (df["Monthly Charges"] > median_charge)
    ).astype(int)

    df["Tenure Bin"] = pd.cut(
        df["Tenure Months"],
        bins=[0, 12, 36, 60, float("inf")],
        labels=[0, 1, 2, 3],
        include_lowest=True,
    ).astype(int)

    df["Sticky Payment"] = (
        (df["Paperless Billing"] == "Yes") &
        (df["Payment Method"].str.contains("automatic", case=False, na=False))
    ).astype(int)

    # ── 7. Customer Charge Index ──────────────────────────────────────────────
    # Inspired by: towardsdatascience.com/predict-customer-churn-with-precision
    # Compute the median monthly charge for each service bundle (Internet +
    # Phone combination) from the full dataset, then express each customer's
    # actual monthly charge as an index against that bundle median.
    # Index > 1.0 = paying above median for their bundle → churn signal
    # Index < 1.0 = paying below median → stickier
    bundle_col = "Internet Service"  # primary bundle driver
    if bundle_col in df.columns and "Phone Service" in df.columns:
        bundle_medians = df.groupby([bundle_col, "Phone Service"])["Monthly Charges"].transform("median")
        df["Charge Index"] = df["Monthly Charges"] / (bundle_medians + 1e-9)
    else:
        df["Charge Index"] = 1.0

    added = ["Charge Ratio", "Avg Monthly Charge", "Services Count",
             "High Risk Flag", "Tenure Bin", "Sticky Payment", "Charge Index"]
    print(f"Feature engineering: +{len(added)} features -> {added}")
    return df


def encode_for_gbm(df: pd.DataFrame):
    """OHE for nominal, LE for binary, ordinal for Contract. For LightGBM + XGBoost."""
    df = df.copy()
    encoders = {}

    # Contract: ordinal (1, 12, 24) — captures commitment hierarchy
    if "Contract" in df.columns:
        df["Contract"] = df["Contract"].map(CONTRACT_ORDINAL).fillna(1).astype(int)
        encoders["Contract"] = CONTRACT_ORDINAL

    for col in BINARY_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    nominal_present = [c for c in NOMINAL_COLS if c in df.columns]
    if nominal_present:
        ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
        ohe_array = ohe.fit_transform(df[nominal_present])
        ohe_cols  = ohe.get_feature_names_out(nominal_present)
        ohe_df    = pd.DataFrame(ohe_array, columns=ohe_cols, index=df.index)
        df = df.drop(columns=nominal_present)
        df = pd.concat([df, ohe_df], axis=1)
        encoders["_ohe"] = ohe

    print(f"Encoded (GBM): {df.shape[1]} total features after OHE expansion")
    return df, encoders



def three_way_split(df: pd.DataFrame, target: str = TARGET_COL):
    """60% train / 20% val / 20% test. Stratified on target."""
    X = df.drop(columns=[target])
    y = df[target]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )

    print(f"Split -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"Churn  -> Train: {y_train.mean():.1%} | Val: {y_val.mean():.1%} | Test: {y_test.mean():.1%}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_and_prepare():
    """
    Returns GBM-encoded data only (XGBoost path).
    CatBoost path removed — not needed for the simplified model.
    """
    raw_df   = load_raw()
    clean_df = clean(raw_df)
    eng_df   = engineer_features(clean_df)
    gbm_df, gbm_encoders = encode_for_gbm(eng_df)
    splits   = three_way_split(gbm_df)

    X_train, X_val, X_test, y_train, y_val, y_test = splits

    return {
        "gbm": {
            "X_train": X_train, "X_val": X_val, "X_test": X_test,
            "encoders": gbm_encoders,
            "feature_names": X_train.columns.tolist(),
        },
        "y_train": y_train,
        "y_val":   y_val,
        "y_test":  y_test,
    }