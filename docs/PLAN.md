# Restructure Plan (slim)

Goal: a module layout, Supabase + Clerk for multi-tenancy, a calibrated model, RAG on
pgvector with evals. **Reuse what exists; add only what's needed.** Each phase ends with
the tests passing and `docker compose up` working.

Ground rules:
- **The FastAPI app stays as it is**: `main.py`, `routes/`, `schemas.py`. The folder moves; its insides don't get re-architected. Only the `db/` calls under it change, from SQLite to Supabase.
- **No new frameworks where pandas, scikit-learn or plain SQL already do the job.** That means no Pandera, dbt, Optuna or Evidently.
- Optional items are marked *(optional)*. Skip them if time is short.

---

## Phase 0 — Prep (20 min)
- [ ] Commit the WIP on `local-docker`, then branch `restructure/monorepo`.
- [ ] Note the baseline: recall 80.5%, AUC 0.85, tests green.

## Phase 1 — Move into the module layout (1.5 h)

Use `git mv` only and make no logic changes in this phase.

| Current | Target |
|---|---|
| `frontend/` | `apps/web/` |
| `api/` (unchanged inside) | `services/api/` |
| `db/*.py` | `services/api/db/` (stays next to the API that uses it) |
| `mcp_server/` | `services/mcp/` |
| `src/data/loader.py` | `packages/data/` |
| `src/model/*`, `src/explainability/*`, `run_pipeline.py` | `packages/ml/` |
| `rag/*` | `packages/rag/` |
| `agents/*`, `guardrails/*` | `packages/agents/` (guardrails as a subfolder) |
| `eval/`, `evaluate.py` | `packages/evals/` |
| `db/schema.sql` | `supabase/migrations/0001_init.sql` |
| `tests/*` | `tests/unit/`, plus a new `tests/integration/` |
| AWS Lambda/SageMaker files (on `main`) | `infra/aws/` (kept, not on the resume) |

- [ ] Fix the imports, Dockerfiles, compose file, `pytest.ini`, `ruff.toml` and CI paths.
- [ ] Keep the requirements files, split by purpose: `requirements-serve.txt` (API runtime) and `requirements-train.txt` (training).

## Phase 2 — Interfaces (1 h)

Three small `Protocol`s, so the agents and routes can be tested with fakes:
- [ ] `Predictor`, wrapping the model (`predict_proba`, `model_version`).
- [ ] `Retriever`, wrapping RAG (`retrieve(query, org_id, k)`).
- [ ] `CustomerRepository`, wrapping the existing functions in `db/customers.py` (`get`, `list`, `features`).
- [ ] Pass them into `build_graph(...)` and into the routes through FastAPI `Depends`.
- [ ] Put fakes in `tests/fakes.py`.

## Phase 3 — Supabase and Clerk (3 h)
- [ ] Translate `schema.sql` to Postgres and add `org_id` to every table. Add new tables `predictions`, `audit_log`, `ab_assignments`, `ab_events`, `tenant_settings` and `rag_documents`.
- [ ] RLS: `org_id = auth.jwt() -> 'o' ->> 'id'`. Check the claim name in your Clerk token first; v1 tokens use `org_id`.
- [ ] Clerk: one organization per client company, with admin, analyst and viewer roles. Register Clerk as a provider in Supabase's Authentication settings.
- [ ] Next.js: Clerk middleware plus an org switcher.
- [ ] FastAPI: one `get_principal` dependency that verifies the JWT against Clerk's JWKS, plus `require_role(...)` on each route. Existing routes keep their code.
- [ ] Clerk webhook route that upserts users and organizations.
- [ ] Swap the internals of the `db/` functions from `sqlite3` to `psycopg`, with the same function names.

## Phase 4 — Data cleaning and transforms (1.5 h, simple)

Build on the existing `loader.py`, which already has `load_raw`, `clean`, `engineer_features` and `three_way_split`.

- [ ] **Input:** `load_raw(path_or_buffer)` also accepts an uploaded CSV, through a new `POST /upload` route on the existing API.
- [ ] **Column mapping:** a plain dict per tenant in `config/tenants/<org>.json`, for example `{"Tenure in Months": "Tenure Months", ...}`, applied with `df.rename(columns=...)` before `clean()`.
- [ ] **Validation:** one `validate(df) -> list[str]` function in pandas that checks:
  - required columns are present
  - dtypes are correct
  - there are no nulls in key fields
  - `tenure >= 0` and `monthly > 0`
  - `customer_id` is unique

  If the list isn't empty, the upload returns a 422 with the list, so bad data fails loudly and readably.
- [ ] **Encoding:** replace the hand-rolled `encode_for_gbm` and the encoding in `predict_tool.py` with one sklearn `ColumnTransformer` inside the model `Pipeline`. This is shared with Phase 5, and training and serving can no longer drift.
- [ ] **Storage:** cleaned rows go to `customers` with their `org_id`.
- [ ] **Analytics:** 3–4 plain SQL views (churn by tenure bucket, contract and payment method; revenue at risk; score histogram), exposed through the existing `metrics` route and shown on the metrics page you're already building.

## Phase 5 — Model (2 h)
- [ ] Logistic-regression baseline vs. XGBoost (the existing `train.py`), both using `StratifiedKFold`. Keep the existing `GridSearchCV`; Optuna isn't needed.
- [ ] Calibrate with `CalibratedClassifierCV(method="isotonic")`, report the Brier score, and recompute the economic threshold on the calibrated probabilities.
- [ ] MLflow (local file store): log runs, register the model, and use the `champion`/`challenger` aliases. Add a short model card as a Markdown artifact.
- [ ] Batch SHAP: `run_pipeline.py` writes scores and SHAP to the `predictions` table, as it already does into `customers`.
- [ ] Retrain script: train a challenger, compare it to the champion on the test split, and promote only if it's better.
- [ ] *(optional)* ONNX export, only if the Vercel bundle size forces it.
- [ ] *(optional)* Drift check: simple per-feature PSI in pandas instead of Evidently.
- [ ] Update the README and resume numbers if recall or AUC change.

## Phase 6 — RAG on pgvector (2 h)

Each document is one churner profile, so there is no chunking. Metadata does the filtering instead.

- [ ] Reuse `rag/build_index.py`, but write to `rag_documents` (embedding plus metadata: `org_id`, contract, tenure bucket, internet, payment, churn reason) instead of Chroma.
- [ ] Add an HNSW index on the embedding and a `tsvector` column with a GIN index.
- [ ] Retrieval runs vector search plus Postgres full-text search. Keep the existing `fusion.py` (RRF) and `rerank.py`, and replace the hand-rolled BM25 with Postgres full-text search.
- [ ] Answers cite `[doc_id]` and pass through the existing guardrails.

## Phase 7 — Evals (2 h)
- [ ] **Synthetic queries:** about 50 queries generated from profile metadata. The relevant set is the source doc plus docs with the same churn reason.
- [ ] **Retrieval:** recall@k, MRR, nDCG, in about 40 lines of Python.
- [ ] **Generation:** faithfulness and citation accuracy with the existing Groq judge in `eval/`.
- [ ] **Ablation table** in the README: dense vs. hybrid, reranker on/off, k = 3/5/10.
- [ ] **Agent checks:** the policy tool is called before any discount is offered, plus guardrail pass rate and cost/latency per run.
- [ ] *(optional)* Hand-label 30–50 outputs and report agreement with the judge.
- [ ] *(optional)* CI eval gate on PRs that touch `packages/rag` or the prompts.

## Phase 8 — A/B module (45 min)
- [ ] Hash-based assignment: `sha256(exp:customer_id)` picks the variant.
- [ ] Log exposures and outcomes to the `ab_*` tables.
- [ ] Power analysis and a two-proportion z-test, in about 30 lines of numpy/scipy.
- [ ] Unit tests for all of it.

## Phase 9 — Tests, CI and deploy (2 h)
- [ ] **Unit tests:** transforms, `validate()`, the risk router (low-risk customers skip agent 2), guardrails, threshold math and A/B.
- [ ] **Integration tests:** routes against local Supabase, and an org A user who can't read org B's rows.
- [ ] **Tooling:** ruff, mypy and pre-commit.
- [ ] **CI:** extend the existing `ci.yml` to run lint, then unit tests, then integration tests (`supabase start`), then the Docker build.
- [ ] **Deploy:** a `deploy.yml` that runs `supabase db push`, deploys to Vercel and runs a smoke test.
- [ ] Before the Vercel step, check its current docs on WebSockets for FastAPI. Keep the serving requirements slim.
- [ ] *(optional)* `retrain.yml` as a manual workflow.

## Phase 10 — FDE polish (1 h)
- [ ] `tenant_settings`: discount ceiling and offer policy. The policy guardrail reads its ceiling from here.
- [ ] Tag LangSmith runs with `org_id`.
- [ ] `docs/RUNBOOK.md`: onboarding a client means mapping JSON, then upload, then score.
- [ ] Refresh `docs/ARCHITECTURE.md`.
- [ ] *(optional)* Demo video.

---

## Order

| Day | Phases |
|---|---|
| Tomorrow | 0 → 1 → 2 → 3 → 4 |
| Next | 5 → 6 → 7 |
| After | 8 → 9 → 10 |

Total is about 17 hours of core work, down from 25+.
