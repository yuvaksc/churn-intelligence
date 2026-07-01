# Architecture

Deep-dive diagrams and design notes. For the overview, see the [README](../README.md).

## Request sequence (WebSocket analysis)

An end-to-end analysis, streamed token-by-token:

```mermaid
sequenceDiagram
    autonumber
    participant U as Browser
    participant API as FastAPI (api)
    participant G as LangGraph war room
    participant MCP as MCP server
    participant LLM as Groq
    participant DB as SQLite

    U->>API: WS /api/analyze/{id}/ws
    API->>DB: load customer features
    API->>G: astream_events(state, v2)

    Note over G: Supervisor → Agent 1 (always first)
    G->>LLM: risk summary (streaming)
    LLM-->>U: token frames (agent1)
    G-->>API: risk_label = HIGH

    Note over G: Supervisor → Agent 2 (gather evidence)
    G->>LLM: evidence report (streaming)
    LLM-->>U: token frames (agent2)

    Note over G: Supervisor → Agent 3 (mitigate)
    G->>MCP: list_tools()
    MCP-->>G: tool schemas
    loop ReAct (bounded)
        G->>LLM: decide next tool / draft
        G->>MCP: call_tool(...)
        MCP-->>G: result (JSON)
    end
    LLM-->>U: token frames (agent3 offer)

    API->>API: guardrails — PII mask · policy ceiling · grounding
    API->>DB: recommendation + audit_log row
    API-->>U: done { trace_id }
```

Low-risk customers short-circuit: the supervisor routes Agent 1 → **FINISH** and Agents 2/3 never run (the UI marks them *skipped*).

## Supervisor state machine

The supervisor is deterministic where the flow is linear and only asks the LLM at the one genuine decision — is the gathered evidence sufficient to draft, or gather once more?

```mermaid
stateDiagram-v2
    [*] --> Diagnose: risk == PENDING
    Diagnose --> Finish: risk == LOW  (risk-gate)
    Diagnose --> Gather: risk == HIGH
    Gather --> Decide: evidence ready
    Decide --> Draft: LLM says "sufficient"
    Decide --> Gather: LLM says "regather" (capped: MAX_AGENT2_RUNS)
    Draft --> Finish: offer ready
    Finish --> [*]
```

A hard `MAX_STEPS` cap and the re-gather cap guarantee termination — a lesson learned in verification, where an unconstrained LLM router looped `agent2` five times until the cap.

## Data model

The `api` process is the **single writer**; the MCP server issues read-only `SELECT`s for its data-backed tools. Raw SQL, no ORM (`sqlite3` + `asyncio.to_thread`), schema in `db/schema.sql`.

```mermaid
erDiagram
    customers {
        INTEGER customer_id PK "pandas test-split index"
        REAL    risk_score
        TEXT    risk_label
        INTEGER true_label
        TEXT    all_features "JSON — full feature dict"
        TEXT    top_shap_drivers "JSON"
    }
    recommendations {
        INTEGER id PK
        TEXT    log_id
        TEXT    customer_id "TEST-{idx}"
        REAL    risk_score
        TEXT    offer_text
        TEXT    status
        TEXT    created_at
    }
    crm_tickets {
        INTEGER id PK
        INTEGER customer_id
        TEXT    subject
        TEXT    category
        TEXT    status
        TEXT    priority
    }
    audit_log {
        INTEGER id PK
        TEXT    trace_id
        TEXT    customer_id
        TEXT    agent_path "JSON"
        TEXT    tools_used "JSON"
        TEXT    retrieved_profiles "JSON"
        TEXT    policy "JSON"
        TEXT    guardrails "JSON"
        INTEGER latency_ms
    }
    meta {
        TEXT key PK
        TEXT value "threshold, seed_version, ..."
    }
```

Tables are linked logically (by `customer_id`) but not FK-constrained — `customers` uses the integer split index, while `recommendations` stores the display id `TEST-{idx}`. `customers` and `crm_tickets` are **seeded** at startup from the model/test-split; `recommendations` and `audit_log` accumulate per run.

## Safety & evaluation flow

```mermaid
flowchart TD
    IN[customer_state + features] --> SCAN[input scan · injection heuristic]
    SCAN --> RUN[war room run]
    RUN --> OFFER[agent 3 offer]
    OFFER --> PII[PII mask]
    PII --> POL[policy-ceiling check]
    POL --> GRD[grounding / citation check]
    GRD --> RESP[response + guardrail report]
    RESP --> AUD[(audit_log)]
    RUN -. offline .-> EVAL[eval.run_eval → LangSmith dataset + experiment]
```

- **Guardrails** run inline on every request (both POST and WS paths) and never block hard (this is an internal analyst tool) — violations are flagged and recorded.
- **Eval** is intentionally **offline** (a separate runner), so the request path stays fast; it scores runs against a LangSmith dataset with deterministic checks + Groq LLM-judges.

## Streaming internals

The WebSocket endpoint (`api/routes/ws.py`) drives `war_room_graph.astream_events(version="v2")` and translates the event stream into UI frames:

| LangGraph event | → WS frame |
|---|---|
| `on_chat_model_stream` (node ∈ agent1/2/3) | `{type: "token", agent, text}` |
| node `on_chain_end` | `{type: "agent_complete", agent, data}` |
| supervisor `on_chain_end` | captured for `agent_path` (not surfaced) |
| stream end | `{type: "done", risk_label, trace_id}` |

Only the three narrative agents stream tokens; the supervisor's structured-output routing and Agent 3's tool-deciding calls carry no useful text. The SSE endpoint (`/stream`) remains as a node-level fallback.
