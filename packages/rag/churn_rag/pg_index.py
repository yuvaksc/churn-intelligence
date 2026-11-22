"""churn_rag.pg_index — embed the corpus and upsert it into pgvector.

    python -m churn_rag.pg_index              # needs DATABASE_URL
    python -m churn_rag.pg_index --limit 200  # quick local run

Idempotent: doc_id is the primary key, so re-running updates rows in place.
"""

from __future__ import annotations

import argparse
import json

from churn_rag.documents import Doc, build_corpus
from churn_rag.embed import get_embedder, to_pgvector

BATCH = 256

_SQL = """insert into rag_documents
          (doc_id, content, churn_label, churn_reason, reason_category, contract, internet_service,
           payment_method, tenure_bucket, monthly_band, metadata, embedding_model, embedding)
          values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::vector)
          on conflict (doc_id) do update set
            content = excluded.content, churn_label = excluded.churn_label,
            churn_reason = excluded.churn_reason, reason_category = excluded.reason_category,
            contract = excluded.contract, internet_service = excluded.internet_service,
            payment_method = excluded.payment_method, tenure_bucket = excluded.tenure_bucket,
            monthly_band = excluded.monthly_band, metadata = excluded.metadata,
            embedding_model = excluded.embedding_model, embedding = excluded.embedding,
            ingested_at = now()"""


def upsert(docs: list[Doc], embedder=None) -> int:
    from psycopg.types.json import Jsonb
    from churn_data.db import tx

    embedder = embedder or get_embedder()
    for start in range(0, len(docs), BATCH):
        batch = docs[start:start + BATCH]
        vectors = embedder.embed([d.content for d in batch])
        params = [[d.doc_id, d.content, d.churn_label, d.churn_reason or None, d.reason_category,
                   d.metadata.get("contract"), d.metadata.get("internet_service"),
                   d.metadata.get("payment_method"), d.metadata.get("tenure_bucket"),
                   d.metadata.get("monthly_band"), Jsonb(d.metadata), embedder.name, to_pgvector(v)]
                  for d, v in zip(batch, vectors)]
        with tx() as c:
            with c.cursor() as cur:
                cur.executemany(_SQL, params)
    return len(docs)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)
    docs = build_corpus()
    if args.limit:
        docs = docs[: args.limit]
    print(json.dumps({"indexed": upsert(docs), "churners": sum(d.churn_label for d in docs)}))


if __name__ == "__main__":
    main()
