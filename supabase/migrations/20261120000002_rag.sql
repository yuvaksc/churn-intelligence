-- 0002 — RAG store: one document per historical (train-split) customer profile.
-- Profiles are short and self-contained, so they are not chunked; metadata does the
-- filtering instead. 384-d embeddings (all-MiniLM-L6-v2) with an HNSW index, plus a
-- generated full-text column for keyword search.

create table public.rag_documents (
  doc_id            text primary key,              -- e.g. "train_1234"
  content           text not null,
  churn_label       smallint not null,
  churn_reason      text,
  reason_category   text,                          -- competitor | price | support | product | other
  contract          text,
  internet_service  text,
  payment_method    text,
  tenure_bucket     text,                          -- 0-11 | 12-23 | 24-47 | 48+
  monthly_band      text,                          -- <35 | 35-70 | 70-95 | 95+
  metadata          jsonb not null default '{}'::jsonb,
  embedding_model   text not null,
  embedding         vector(384) not null,
  fts               tsvector generated always as (
                      to_tsvector('english', content || ' ' || coalesce(churn_reason, ''))
                    ) stored,
  ingested_at       timestamptz not null default now()
);

create index rag_documents_embedding_hnsw on public.rag_documents
  using hnsw (embedding vector_cosine_ops) with (m = 16, ef_construction = 64);
create index rag_documents_fts_gin   on public.rag_documents using gin (fts);
create index rag_documents_label_idx on public.rag_documents (churn_label, contract, internet_service);

alter table public.rag_documents enable row level security;

-- Hybrid retrieval in one round trip, three signals fused with Reciprocal Rank Fusion:
--   dense       cosine similarity over the HNSW index (meaning of the profile text)
--   sparse      Postgres full-text search, the query's words OR-ed together
--   structured  nearest neighbours on the numbers text search cannot see: tenure months,
--               monthly charge, payment method (|Δtenure|/12 + |Δcharge|/15 + 0.5 if the
--               payment method differs). Off when query_tenure is null.
--   RRF score = Σ 1 / (rrf_k + rank) over the signals a document appears in.
-- Optional metadata pre-filters (contract, internet type) restrict candidates first.
-- Measured on 40 HIGH-risk customers, adding the structured signal moved nDCG@5 from
-- 0.31 to 0.63 and cut the top-5 tenure gap from 13 to 6 months (README).
create or replace function public.hybrid_search(
  query_embedding vector(384),
  query_text      text,
  match_count     int     default 30,
  churners_only   boolean default true,
  rrf_k           int     default 60,
  use_dense       boolean default true,
  use_sparse      boolean default true,
  filter_contract text    default null,
  filter_internet text    default null,
  query_tenure    int     default null,
  query_charge    double precision default null,
  query_payment   text    default null
)
returns table (
  doc_id text, content text, churn_label smallint, churn_reason text, reason_category text,
  metadata jsonb, similarity double precision, dense_rank bigint, sparse_rank bigint,
  structured_rank bigint, rrf_score double precision
)
language sql stable
as $$
  with q as (
    select nullif(replace(plainto_tsquery('english', query_text)::text, '&', '|'), '')::tsquery as tsq
  ),
  candidates as (
    select d.* from public.rag_documents d
    where (not churners_only or d.churn_label = 1)
      and (filter_contract is null or d.contract = filter_contract)
      and (filter_internet is null or d.internet_service = filter_internet)
  ),
  dense as (
    select c.doc_id, row_number() over (order by c.embedding <=> query_embedding) as rnk
    from candidates c
    where use_dense
    order by c.embedding <=> query_embedding
    limit match_count
  ),
  sparse as (
    select c.doc_id, row_number() over (order by ts_rank_cd(c.fts, q.tsq) desc) as rnk
    from candidates c, q
    where use_sparse and q.tsq is not null and c.fts @@ q.tsq
    order by ts_rank_cd(c.fts, q.tsq) desc
    limit match_count
  ),
  structured as (
    select s.doc_id, row_number() over (order by s.dist) as rnk
    from (
      select c.doc_id,
             abs((c.metadata->>'tenure_months')::int - query_tenure) / 12.0
           + abs((c.metadata->>'monthly_charges')::float - coalesce(query_charge, 0)) / 15.0
           + case when query_payment is null or c.payment_method = query_payment then 0 else 0.5 end as dist
      from candidates c
      where query_tenure is not null
    ) s
    order by s.dist
    limit match_count
  ),
  fused as (
    select coalesce(dense.doc_id, sparse.doc_id, structured.doc_id) as doc_id,
           dense.rnk as dense_rank, sparse.rnk as sparse_rank, structured.rnk as structured_rank,
           coalesce(1.0 / (rrf_k + dense.rnk), 0) + coalesce(1.0 / (rrf_k + sparse.rnk), 0)
         + coalesce(1.0 / (rrf_k + structured.rnk), 0) as rrf_score
    from dense
    full outer join sparse on dense.doc_id = sparse.doc_id
    full outer join structured on structured.doc_id = coalesce(dense.doc_id, sparse.doc_id)
  )
  select d.doc_id, d.content, d.churn_label, d.churn_reason, d.reason_category, d.metadata,
         1 - (d.embedding <=> query_embedding) as similarity,
         f.dense_rank, f.sparse_rank, f.structured_rank, f.rrf_score::double precision
  from fused f
  join public.rag_documents d on d.doc_id = f.doc_id
  order by f.rrf_score desc
  limit match_count
$$;
