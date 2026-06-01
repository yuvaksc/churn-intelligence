"""
rag/build_index.py — One-time ChromaDB indexing.

Run once from project root:
    python rag/build_index.py

Creates two collections in data/chroma_db/:
  churn_profiles  — 4225 training customer summaries (pre-encoding, human-readable)
                    Used by Agent 2 to find similar historical customers
  churn_reasons   — 1869 churn reason strings from churn_reasons_rag.csv
                    Used by Agent 2 to surface why similar customers left

Key design:
  - Uses eng_df (pre-encoding) so text is human-readable ("Fiber optic" not "1")
  - Reconstructs the EXACT same train/val/test split (random_state=42)
    so no test-set data leaks into the RAG corpus
  - Embeddings: sentence-transformers all-MiniLM-L6-v2 (bundled with chromadb)
  - Batch size 500 to stay within ChromaDB memory limits
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from data.loader import (
    load_raw, clean, engineer_features,
    encode_for_gbm, three_way_split,
)

CHROMA_DIR = Path("data/chroma_db")
BATCH_SIZE  = 500


# ── Text builders ─────────────────────────────────────────────────────────────

def build_profile_text(row: pd.Series, label: int) -> str:
    """
    Convert one pre-encoded training row into a searchable text document.
    Keeps all values in their original string/numeric form for readability.
    """
    outcome = "Churned" if label == 1 else "Stayed"
    return (
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
        f"Senior Citizen: {'Yes' if row.get('Senior Citizen', 0) == 1 else 'No'} | "
        f"Partner: {row.get('Partner', 'Unknown')} | "
        f"Dependents: {row.get('Dependents', 'Unknown')} | "
        f"Services Count: {int(row.get('Services Count', 0))} | "
        f"High Risk Flag: {'Yes' if row.get('High Risk Flag', 0) == 1 else 'No'} | "
        f"Tenure Bin: {int(row.get('Tenure Bin', 0))} | "
        f"Sticky Payment: {'Yes' if row.get('Sticky Payment', 0) == 1 else 'No'} | "
        f"Outcome: {outcome}"
    )


def build_profile_metadata(row: pd.Series, label: int) -> dict:
    """
    Structured metadata for ChromaDB filtering (e.g. churners_only=True).
    Only scalar types — ChromaDB does not accept lists or nested dicts.
    """
    return {
        "churn_label":     label,
        "contract":        str(row.get("Contract", "")),
        "internet_service":str(row.get("Internet Service", "")),
        "tenure_months":   int(row.get("Tenure Months", 0)),
        "monthly_charges": float(row.get("Monthly Charges", 0)),
        "services_count":  int(row.get("Services Count", 0)),
        "high_risk_flag":  int(row.get("High Risk Flag", 0)),
        "tenure_bin":      int(row.get("Tenure Bin", 0)),
    }


# ── Collection builders ───────────────────────────────────────────────────────

def build_churn_profiles(client, eng_df: pd.DataFrame, train_idx, y_train: pd.Series):
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

    try:
        client.delete_collection("churn_profiles")
        print("  Dropped existing churn_profiles collection")
    except Exception:
        pass

    collection = client.create_collection("churn_profiles", embedding_function=ef)

    train_rows = eng_df.loc[train_idx]

    documents, metadatas, ids = [], [], []

    for row_idx, row in train_rows.iterrows():
        label = int(y_train.loc[row_idx])
        documents.append(build_profile_text(row, label))
        metadatas.append(build_profile_metadata(row, label))
        ids.append(f"train_{row_idx}")

    # Insert in batches
    for start in range(0, len(documents), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(documents))
        collection.add(
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
        )
        print(f"  churn_profiles: indexed {end}/{len(documents)} rows")

    churn_count = sum(1 for m in metadatas if m["churn_label"] == 1)
    print(f"  churn_profiles: {collection.count()} total  "
          f"({churn_count} churners / {len(documents) - churn_count} retained)")
    return collection


def build_churn_reasons(client):
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

    try:
        client.delete_collection("churn_reasons")
        print("  Dropped existing churn_reasons collection")
    except Exception:
        pass

    collection = client.create_collection("churn_reasons", embedding_function=ef)

    rag_df = pd.read_csv("data/raw/churn_reasons_rag.csv")

    documents, metadatas, ids = [], [], []

    for i, row in rag_df.iterrows():
        reason = str(row.get("Churn Reason", "")).strip()
        if reason and reason.lower() != "nan":
            documents.append(reason)
            metadatas.append({
                "customer_id":  str(row.get("CustomerID", f"unk_{i}")),
                "churn_value":  int(row.get("Churn Value", 1)),
            })
            ids.append(f"reason_{row.get('CustomerID', i)}")

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print(f"  churn_reasons:  {collection.count()} reasons indexed")
    return collection


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import chromadb

    print("=" * 55)
    print("  Building ChromaDB RAG Index")
    print("=" * 55)

    # Reconstruct exact same pipeline state as training (random_state=42)
    print("\nReproducing training split...")
    raw_df   = load_raw()
    clean_df = clean(raw_df)
    eng_df   = engineer_features(clean_df)

    # encode_for_gbm gives us the same indices as training
    gbm_df, _ = encode_for_gbm(eng_df)
    X_train, X_val, X_test, y_train, y_val, y_test = three_way_split(gbm_df)

    print(f"  Training rows for RAG: {len(X_train)}")
    print(f"  Test rows excluded:    {len(X_test)}  (no leakage)")

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    print("\n[1/2] Building churn_profiles collection...")
    build_churn_profiles(client, eng_df, X_train.index, y_train)

    print("\n[2/2] Building churn_reasons collection...")
    build_churn_reasons(client)

    print(f"\nChromaDB persisted → {CHROMA_DIR.resolve()}")
    print("Re-run only if training data changes.")
    print("=" * 55)


if __name__ == "__main__":
    main()