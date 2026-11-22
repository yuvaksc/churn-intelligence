# Churn Intelligence — Multi-Agent Retention Platform

A telecom churn model turned into an analyst tool. A **calibrated XGBoost** model scores each
customer; for high-risk customers a **LangGraph multi-agent "war room"** diagnoses the risk,
gathers evidence from similar past churners (**hybrid RAG on pgvector**), and drafts a
policy-safe retention offer (a **ReAct agent calling MCP tools**), streaming each agent's
reasoning to the UI over **WebSockets**. Every stage is measured: model metrics with a
baseline, and every analysis traced in LangSmith and recorded as experiments scored on
retrieval metrics, guardrail checks and LLM-as-judge quality.

> FastAPI · LangGraph · Groq · MCP · Supabase Postgres + pgvector · MLflow · Next.js · Vercel · GitHub Actions

[![CI](https://github.com/yuvaksc/churn-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/yuvaksc/churn-intelligence/actions/workflows/ci.yml)

---

## How it works in one minute

```mermaid
flowchart LR
    U([Browser]) -->|HTTP + WebSocket| WEB[apps/web<br/>Next.js]
    WEB --> API[services/api<br/>FastAPI]
    API --> G{{LangGraph war room}}
    G -->|score| ML[(champion model<br/>calibrated XGBoost)]
    G -->|evidence| PG[(Supabase Postgres<br/>pgvector + full-text)]
    G -->|tools| MCP[services/mcp<br/>policy · competitors · CRM]
    MCP --> PG
    G -->|LLM| GROQ[[Groq gpt-oss-120b]]
    API -.->|traces| LS[[LangSmith]]
```

1. **Agent 1 · Diagnostic** scores the customer with the calibrated model and explains the top
   SHAP drivers in plain language. LOW-risk customers stop here (no LLM cost).
2. **Agent 2 · Evidence** retrieves similar historical churners and their documented reasons,
   and writes a report where every claim cites a retrieved customer (`[train_1234]`).
3. **Agent 3 · Mitigation** gathers context through MCP tools (discount policy, competitor offers,
   account history, open tickets, prior offers) and drafts a HOOK / OFFER / URGENCY offer.
4. **Guardrails** mask PII, enforce the policy's discount ceiling, check grounding and citations,
   and scan inputs for prompt injection.

---

## Repository layout

```
apps/web/                   Next.js dashboard + live-streaming war room UI
services/api/               FastAPI: routes → interfaces (repositories, predictor, retriever)
services/mcp/               MCP server: retention policy, competitor intel, CRM tools
packages/
  data/churn_data/          loader, feature engineering, Postgres repositories, CRM rules
  ml/churn_ml/              pipeline, training + calibration, economics, MLflow registry,
                            predictor (serving), scoring (batch SHAP + seeding)
  rag/churn_rag/            corpus, embeddings, pgvector index, hybrid retriever, reranker
  agents/churn_agents/      LangGraph graph, supervisor, 3 agents, prompts, guardrails, tools
  evals/churn_evals/        evaluators + per-analysis LangSmith experiments
supabase/migrations/        the database schema (applied by Supabase and by docker compose)
tests/unit · tests/integration
.github/workflows/          ci · deploy · retrain
```

Each package is importable as `churn_*`. Agents and routes depend on **interfaces**
(`Predictor`, `Retriever`, `CustomerRepository`, `ToolGateway`), never on concrete clients,
which is why the entire war room runs on in-memory fakes in the unit tests.

---

## Module by module

### ML and MLOps (`packages/ml`)

```
telco data → 60/20/20 stratified split
  → one sklearn Pipeline: FeatureEngineer (medians learned at fit) → ColumnTransformer → model
  → baseline: logistic regression   vs   XGBoost tuned with 5-fold stratified CV
  → isotonic calibration (Brier score reported)
  → decision threshold chosen from retention economics on calibrated probabilities
  → MLflow: register as @challenger → promote to @champion only if it beats the champion
  → scoring job: every customer's score + top-5 SHAP drivers → Postgres
```

| file | role |
|---|---|
| `model.py` | the sklearn Pipeline, used for training and serving |
| `training.py` | splits, baseline vs tuned XGBoost, calibration, threshold |
| `economics.py` | the retention economics behind the threshold (also used by `/api/metrics`) |
| `registry.py` | MLflow tracking + registry, champion / challenger, export `champion.joblib` |
| `retrain.py` | one command: train → register → compare → promote |
| `predictor.py` | the `Predictor` interface + the loader of `champion.joblib` used by Agent 1 |
| `scoring.py` | scores every customer + SHAP drivers into Postgres (`--seed` loads the customers first) |

* **No train/serve skew.** Preprocessing lives inside the pipeline and is saved with it; serving
  never re-implements encoding.
* **Calibrate, then threshold.** Contacting a customer is worth it when
  `p · save-rate · monthly · LTV > offer cost`. That only works if `p` is a real probability.
* **SHAP in batch, not per request.** The API serves a precomputed explanation.
* `python -m churn_ml.retrain` trains, compares and promotes; the `retrain` workflow runs it weekly.

### RAG (`packages/rag`)

* **Corpus:** one document per train-split customer; profiles are short, so they are **not chunked**.
  Each carries metadata (churn reason and its category, contract, internet type, payment method,
  tenure bucket, charge band). Test customers are never indexed.
* **Retrieval:** a single Postgres function `hybrid_search` pre-filters on the customer's
  contract and internet type, then fuses **three ranked lists with Reciprocal Rank Fusion**:
  **dense** (pgvector cosine, HNSW), **sparse** (full-text keyword match) and **structured**
  (nearest neighbours on tenure, monthly charge and payment method, the numbers text search
  cannot tell apart). A **cross-encoder** reranks the fused list. The query is the customer's
  profile only.
* **Generation:** Agent 2 must cite `[doc_id]`s; a guardrail rejects any citation that was not retrieved.

### Agents and prompts (`packages/agents`)

All prompts live in `churn_agents/prompts.py` and share one structure: **ROLE → TASK →
STEPS (chain of thought) → CONSTRAINTS → OUTPUT FORMAT → EXAMPLE**. Fixed instructions go in the
system message; per-run data goes in the user message, so instructions and data never mix. The
model works through the STEPS before answering but outputs only the final answer, because agent
text streams to the UI. Agent 3 is token-budgeted: all tools requested in one turn, compact tool
results, at most three rounds.

### Data (`packages/data`, `supabase/`)

Postgres with raw SQL (psycopg 3). Tables: `customers`, `predictions`, `recommendations`,
`crm_tickets`, `meta`, `rag_documents`. Row-level
security is enabled with no policies, so Supabase's public API cannot read anything; only the
backend's direct connection can.

---

## Results

All numbers come from the code in this repo; the commands reproduce them.

### Model — held-out test set (1,409 customers)

| | Logistic regression (baseline) | **XGBoost (tuned, calibrated)** |
|---|---|---|
| ROC-AUC | 0.845 | **0.849** |
| PR-AUC | 0.611 | **0.640** |
| Brier (uncalibrated → isotonic) | 0.164 → 0.139 | 0.159 → **0.135** |
| Recall @ economic threshold | 0.963 | 0.936 |
| Precision @ economic threshold | 0.412 | 0.435 |

Both models share the pipeline and calibration, so the lift is the model's alone: a modest
but real gain over a strong linear baseline. The threshold (0.13) maximises retention net value
under the assumptions in `churn_ml/economics.py` (30% save rate, 24-month LTV, ~$62 per offer):
offers are cheap relative to a saved customer, so flagging ~57% of the base is optimal.

### Evaluation — every analysis is traced and scored in LangSmith

**Tracing.** Each analysis is **one trace**, `war-room · customer <id>`, in the `churn` project
(tagged with the prompt version). Its waterfall shows every step: supervisor → Agent 1
(`churn_model.predict`, its LLM call) → Agent 2 (`hybrid_search` retriever span with the ranked
documents, its LLM call) → Agent 3 (each MCP tool call as a `tool` span, each ReAct LLM call).

**Experiments.** When the analysis finishes (after the UI has its answer),
`churn_evals/live.py` records it as LangSmith experiments, scored by `churn_evals/evaluators.py`:

| dataset | experiment | metrics |
|---|---|---|
| `churn-warroom` | `warroom-<customer>-prompt-<v>` (every analysis) | routing, policy tool before offer, guardrails, citations, **LLM-as-judge** quality (0–1) with reasoning |
| `churn-rag-golden` | `rag-<customer>-prompt-<v>` (HIGH risk, where Agent 2 ran) | recall@k, precision@k, nDCG@k (k = 3, 5, 10), MRR, citation accuracy, **LLM-as-judge** faithfulness + answer relevance |

Both datasets are small reference sets (8 examples each) that the experiments attach to; the
same customer always lands on the same example, so **to compare prompt versions, bump
`PROMPT_VERSION` in `prompts.py`, re-run the same customer, and compare the two experiments
side by side**. The RAG relevance labels are computed for the analysed customer from document
metadata: grade 2 = a churner with the same contract, internet type, tenure bucket, charge band
and payment method; grade 1 = all but payment. The labels are strict (a 12-month customer is
not "relevant" to an 11-month one), so compare scores between versions and configurations.

**Retrieval: how the pipeline was fixed** (40 HIGH-risk customers; the last two columns are the
average gap between the customer and its top 5 retrieved churners, which does not depend on the
relevance labels at all):

| configuration | nDCG@5 | P@5 | MRR | tenure gap | charge gap |
|---|---|---|---|---|---|
| before: query = LLM risk summary + profile; dense + full-text; cross-encoder | 0.10 | 0.16 | 0.33 | 17.2 mo | $14.6 |
| query = profile only | 0.31 | 0.40 | 0.64 | 12.8 mo | $9.8 |
| full-text only / dense only (profile query, no reranker) | 0.13 / 0.27 | 0.15 / 0.43 | 0.28 / 0.46 | 18.8 / 16.0 mo | $17.4 / $12.2 |
| **+ structured signal, 3-way RRF, cross-encoder (shipped)** | **0.63** | **0.76** | **0.92** | **5.6 mo** | **$4.5** |
| same without the cross-encoder | 0.65 | 0.63 | 0.87 | 6.2 mo | $6.6 |

Two root causes. (1) Agent 2 searched with the LLM's prose risk summary prepended to the profile,
so the prose dominated both the embedding and the keyword match; the offline evals had used the
profile alone and never saw it. (2) Customer profiles are mostly numbers, and neither embeddings
nor keyword search can tell "Tenure 1 months" from "Tenure 13 months": a 1-month, $71 customer
was matched with 13-to-70-month, $95-110 churners. A structured nearest-neighbour signal fused
with the text signals fixes that, and the cross-encoder still adds precision on top. Postgres
full-text search is the weakest signal (it ranks without IDF, unlike BM25), so it carries little
weight on its own but helps in the fusion. Caveat: the relevance labels use tenure and charge
buckets, so the structured signal is partly aligned with the metric; the gap columns show the
gain is real regardless.

**Prompt v4 results** (development runs: 12 evidence reports, 8 full war rooms):
faithfulness 0.83, answer relevance 1.00, citation accuracy 1.00; routing 8/8, policy tool
before offer 5/5, guardrails 5/5, war-room quality 0.75. Both unfaithful reports misjudged a
monthly-charge range at its edge (e.g. "matches" for a $57.45 customer when the range tops out
at $57.15), which is the next prompt change to try.

How the prompt got to v4 (each row one change, measured on the same 12 queries):

| prompt version | faithfulness | relevance | what changed / what the eval showed |
|---|---|---|---|
| v0 original (unstructured) | 0.83 | 1.00 | baseline |
| v1 structured (role, steps, constraints, format, example) | 0.67 | 0.75 | over-refused: "insufficient evidence" despite good evidence |
| v2 | 0.67 | 0.92 | refusal only when nothing matches; exact ranges |
| v3 | 0.83 | 0.83 | ranges computed in code; no "most common" when reasons tie |
| v4 (judge sees the full prompt context) | 1.00 | 0.92 | judge fixed to see everything the generator saw |

Two lessons the eval made visible: stricter refusal rules cause over-refusal, and LLMs should
not do arithmetic the code can do. Caveats: small samples, and the judge is the same model as the
generator (self-grading bias); spot-check its verdicts in a LangSmith annotation queue.

---

## Run it locally

Prerequisites: Python 3.12, Node 22, Docker, a [Groq API key](https://console.groq.com) in `.env`
(see `.env.example`).

```bash
pip install -r requirements.txt -r requirements_agents.txt -r requirements_api.txt
docker compose up -d db                      # Postgres + pgvector on :5433, migrations applied
export DATABASE_URL=postgresql://postgres:postgres@localhost:5433/churn
export PYTHONPATH="packages/data:packages/ml:packages/rag:packages/agents:packages/evals"

python -m churn_ml.retrain --promote force   # train, register in MLflow, export the champion
python -m churn_ml.scoring --seed            # customers + scores + SHAP + CRM tickets
python -m churn_rag.pg_index                 # embed + index the RAG corpus

python scripts/dev_api.py                    # API on :8000 (+ MCP server on :8001)
cd apps/web && npm ci && npm run build && npx next start -p 3000
```

Open http://localhost:3000, pick a HIGH-risk customer, and click **Run Analysis**.
Or run everything in containers: `docker compose up -d --build` (see the header of
`docker-compose.yml` for the one-time train / seed / index commands).

## Deploy (Supabase + Vercel)

1. **Supabase:** create a project and run the two files in `supabase/migrations/` (in order) in
   the SQL Editor, or `npx supabase db push`. From **Connect**, copy the **Session pooler** URI
   (port 5432, for loading data from your machine) and the **Transaction pooler** URI (port
   6543, for Vercel).
2. **Data:** with `DATABASE_URL` set to the session pooler URI, run `churn_ml.scoring --seed` and
   `churn_rag.pg_index` once from your machine. `models/champion.joblib` is the only tracked
   model file and ships with the deploy.
3. **Vercel:** import the repo (root directory: the repo root). `vercel.json` defines two
   services (`web` → `apps/web`, `api` → `services/api/vercel_entry.py`) behind one domain, so
   the web app calls the API same-origin. Set `DATABASE_URL` (transaction pooler), `GROQ_API_KEY`,
   `EMBEDDER=fastembed` (ONNX instead of torch, keeps the Python bundle small; same results),
   and optionally the `LANGSMITH_*` variables.
4. **CI/CD:** add the secrets listed at the top of `.github/workflows/deploy.yml`; merges to `main`
   then apply migrations, deploy, and run `scripts/smoke_test.py` against the live URL.

The public demo has no login, so `/api/analyze` is rate-limited per IP (`ANALYSIS_LIMIT_PER_HOUR`,
default 20) to protect the LLM budget.

## Testing and CI

```bash
pip install -r requirements-test.txt
pytest tests/unit                       # no network, no database: everything faked
docker run -d -p 54329:5432 -e POSTGRES_PASSWORD=postgres pgvector/pgvector:pg16
TEST_DATABASE_URL=postgresql://postgres:postgres@localhost:54329/postgres pytest tests/integration
```

| Workflow | Trigger | What |
|---|---|---|
| `ci.yml` | every PR / push | ruff + mypy → unit tests (coverage floor) → integration tests on pgvector Postgres → web typecheck + build → Docker builds (GHCR on push) → Vercel preview (PRs) |
| `deploy.yml` | merge to `main` | `supabase db push` → Vercel production → smoke test |
| `retrain.yml` | weekly / manual | train → challenger vs champion → promote → re-score customers |

What the tests cover: the war-room graph on fakes (low-risk customers skip agents 2 and 3; the
policy tool runs before any offer; backstops), guardrails, prompt structure, threshold economics,
calibration-aware serving, the eval metrics, the REST and WebSocket routes, rate limiting, and
on real Postgres: repositories, hybrid search with filters and the structured signal, and the
RLS lockout of public roles.

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | status, customers, high-risk count, threshold, model version, RAG documents |
| `GET` | `/api/customers` | risk-sorted list (`limit`, `offset`, `risk_only`) |
| `GET` | `/api/customers/{id}` | features + precomputed SHAP drivers |
| `WS` | `/api/analyze/{id}/ws` | run the war room, stream tokens and agent results |
| `GET` | `/api/logs` | drafted retention offers |
| `GET` | `/api/metrics` | model metrics + modelled business value |
