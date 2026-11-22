"""churn_rag.documents — ingest and clean the RAG corpus (one doc per profile).

Profiles are short and self-contained, so they are NOT chunked; instead each
carries metadata that retrieval and evals can filter or grade on:

    doc_id, churn_label, churn_reason, reason_category, contract, internet_service,
    payment_method, tenure_bucket, monthly_band, source, embedding_model

Only TRAIN-split customers are indexed, and a churner's reason is metadata, never
a model feature — the same no-leakage rule as before.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from churn_data.loader import clean, encode_for_gbm, engineer_features, load_raw, three_way_split

REASON_CATEGORY = {
    "Competitor made better offer": "competitor", "Competitor had better devices": "competitor",
    "Competitor offered more data": "competitor", "Competitor offered higher download speeds": "competitor",
    "Price too high": "price", "Extra data charges": "price", "Long distance charges": "price",
    "Lack of affordable download/upload speed": "price",
    "Attitude of support person": "support", "Attitude of service provider": "support",
    "Poor expertise of phone support": "support", "Poor expertise of online support": "support",
    "Lack of self-service on Website": "support",
    "Network reliability": "product", "Product dissatisfaction": "product",
    "Service dissatisfaction": "product", "Limited range of services": "product",
    "Moved": "other", "Deceased": "other", "Don't know": "other",
}

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_PHONE = re.compile(r"\+?\d[\d\s().-]{7,}\d")
_WS = re.compile(r"\s+")


@dataclass
class Doc:
    doc_id: str
    content: str
    churn_label: int
    churn_reason: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def reason_category(self) -> str | None:
        return REASON_CATEGORY.get(self.churn_reason, "other") if self.churn_reason else None


def tenure_bucket(months: int) -> str:
    return "0-11" if months < 12 else "12-23" if months < 24 else "24-47" if months < 48 else "48+"


def monthly_band(charge: float) -> str:
    return "<35" if charge < 35 else "35-70" if charge < 70 else "70-95" if charge < 95 else "95+"


def clean_text(text: str) -> str:
    """Normalize whitespace and scrub anything that looks like PII."""
    text = _EMAIL.sub("[email]", text)
    text = _PHONE.sub("[phone]", text)
    return _WS.sub(" ", text).strip()


def profile_text(row: pd.Series, label: int) -> str:
    """One train-split customer as a searchable "Key: value | ..." document."""
    def yes(v) -> str:
        return "Yes" if str(v).strip() in ("Yes", "1", "1.0") else "No"
    text = (
        f"Tenure {int(row.get('Tenure Months', 0))} months | "
        f"Contract: {row.get('Contract', 'Unknown')} | "
        f"Monthly Charges: ${float(row.get('Monthly Charges', 0)):.2f} | "
        f"Total Charges: ${float(row.get('Total Charges', 0)):.2f} | "
        f"Internet Service: {row.get('Internet Service', 'Unknown')} | "
        f"Phone Service: {row.get('Phone Service', 'Unknown')} | "
        f"Online Security: {row.get('Online Security', 'Unknown')} | "
        f"Online Backup: {row.get('Online Backup', 'Unknown')} | "
        f"Tech Support: {row.get('Tech Support', 'Unknown')} | "
        f"Streaming TV: {row.get('Streaming TV', 'Unknown')} | "
        f"Streaming Movies: {row.get('Streaming Movies', 'Unknown')} | "
        f"Device Protection: {row.get('Device Protection', 'Unknown')} | "
        f"Payment Method: {row.get('Payment Method', 'Unknown')} | "
        f"Paperless Billing: {row.get('Paperless Billing', 'Unknown')} | "
        f"Senior Citizen: {yes(row.get('Senior Citizen', 'No'))} | "
        f"Partner: {row.get('Partner', 'Unknown')} | "
        f"Dependents: {row.get('Dependents', 'Unknown')} | "
        f"Services Count: {int(row.get('Services Count', 0))} | "
        f"High Risk Flag: {yes(row.get('High Risk Flag', 0))} | "
        f"Tenure Bin: {int(row.get('Tenure Bin', 0))} | "
        f"Sticky Payment: {yes(row.get('Sticky Payment', 0))} | "
        f"Outcome: {'Churned' if label == 1 else 'Stayed'}"
    )
    return clean_text(text)


def build_corpus(include_reason: bool = True) -> list[Doc]:
    """include_reason appends a churner's documented reason to the text (the
    'profile+reason' template, which won the retrieval ablation)."""
    raw = load_raw()
    eng = engineer_features(clean(raw))
    gbm, _ = encode_for_gbm(eng)
    X_train, _, _, y_train, _, _ = three_way_split(gbm)

    docs, seen = [], set()
    for idx in X_train.index:
        row, label = eng.loc[idx], int(y_train.loc[idx])
        reason = ""
        if label == 1 and "Churn Reason" in raw.columns:
            r = str(raw.loc[idx, "Churn Reason"]).strip()
            reason = "" if r.lower() == "nan" else r
        content = profile_text(row, label)
        if include_reason and reason:
            content = f"{content} | Churn reason: {reason}"
        if (content, reason) in seen:          # exact duplicates add nothing to retrieval
            continue
        seen.add((content, reason))
        docs.append(Doc(
            doc_id=f"train_{idx}", content=content, churn_label=label, churn_reason=reason,
            metadata={
                "contract": str(row["Contract"]), "internet_service": str(row["Internet Service"]),
                "payment_method": str(row["Payment Method"]),
                "tenure_months": int(row["Tenure Months"]), "tenure_bucket": tenure_bucket(int(row["Tenure Months"])),
                "monthly_charges": float(row["Monthly Charges"]),
                "monthly_band": monthly_band(float(row["Monthly Charges"])),
                "services_count": int(row["Services Count"]), "high_risk_flag": int(row["High Risk Flag"]),
            },
        ))
    return docs
