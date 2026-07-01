-- db/schema.sql — applied idempotently at startup (db.connection.init_db).
-- Pure SQL, no ORM. Shapes mirror api/schemas.py so the API/frontend types
-- are unchanged by the migration.

-- Pre-scored test customers served to the dashboard (read-only at runtime,
-- (re)built by db.customers.seed_customers from the model + test split).
CREATE TABLE IF NOT EXISTS customers (
    customer_id      INTEGER PRIMARY KEY,   -- pandas test-split index
    risk_score       REAL    NOT NULL,
    risk_label       TEXT    NOT NULL,      -- "HIGH" | "LOW"
    contract         TEXT    NOT NULL,
    monthly_charges  REAL    NOT NULL,
    tenure_months    INTEGER NOT NULL,
    internet_service TEXT    NOT NULL,
    services_count   INTEGER NOT NULL,
    high_risk_flag   INTEGER NOT NULL,
    true_label       INTEGER,               -- ground truth (nullable)
    all_features     TEXT    NOT NULL,       -- JSON: full pre-encoding feature dict
    top_shap_drivers TEXT    NOT NULL        -- JSON: list[ShapDriver]
);
CREATE INDEX IF NOT EXISTS idx_customers_risk  ON customers(risk_score DESC);
CREATE INDEX IF NOT EXISTS idx_customers_label ON customers(risk_label);

-- Retention/CRM actions, accumulated across runs (1:1 with RetentionLogEntry).
CREATE TABLE IF NOT EXISTS recommendations (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    log_id         TEXT    NOT NULL,
    customer_id    TEXT    NOT NULL,        -- e.g. "TEST-1234"
    risk_score     REAL    NOT NULL,
    offer_text     TEXT    NOT NULL,
    contract_type  TEXT    NOT NULL,
    monthly_charge REAL    NOT NULL,
    timestamp      TEXT    NOT NULL,
    status         TEXT    NOT NULL,
    assigned_to    TEXT    NOT NULL,
    created_at     TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_rec_customer ON recommendations(customer_id);
CREATE INDEX IF NOT EXISTS idx_rec_status   ON recommendations(status);

-- Synthetic CRM support tickets (read-only at runtime; seeded by db.crm.seed_crm
-- from customer features). Exposed to agent3 via the get_open_tickets MCP tool.
CREATE TABLE IF NOT EXISTS crm_tickets (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER NOT NULL,   -- pandas test-split index (matches customers.customer_id)
    subject     TEXT    NOT NULL,
    category    TEXT    NOT NULL,
    status      TEXT    NOT NULL,   -- "Open" | "Resolved"
    priority    TEXT    NOT NULL,   -- "Low" | "Medium" | "High"
    opened_at   TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tickets_customer ON crm_tickets(customer_id);

-- Small key/value store (threshold, seed_version, …).
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Append-only audit trail: one row per war-room analysis (Step 5).
CREATE TABLE IF NOT EXISTS audit_log (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id            TEXT    NOT NULL,
    customer_id         TEXT    NOT NULL,
    customer_state      TEXT,
    risk_label          TEXT,
    risk_score          REAL,
    agent_path          TEXT,   -- JSON: supervisor routing decisions
    tools_used          TEXT,   -- JSON: MCP tools the LLM called
    retrieved_profiles  TEXT,   -- JSON: similar-profile summaries + scores
    churn_reasons_count INTEGER,
    policy              TEXT,   -- JSON
    retention_offer     TEXT,
    crm_log_id          TEXT,
    guardrails          TEXT,   -- JSON: output guardrail report
    input_scan          TEXT,   -- JSON: input guardrail report
    latency_ms          INTEGER,
    created_at          TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_audit_trace    ON audit_log(trace_id);
CREATE INDEX IF NOT EXISTS idx_audit_customer ON audit_log(customer_id);
