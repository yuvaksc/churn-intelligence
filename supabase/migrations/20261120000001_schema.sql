-- 0001 — core schema: customers, scores, offers, CRM, model metadata.
--
-- The API and batch jobs connect directly to Postgres (DATABASE_URL). Every table has
-- row-level security ENABLED with NO policies: Supabase's public REST API (anon /
-- authenticated keys) can read nothing, while the backend's direct connection works.

create extension if not exists vector;

-- Customers served to the dashboard (seeded from the held-out test split).
create table public.customers (
  customer_id       bigint primary key,           -- pandas test-split index
  risk_score        double precision,
  risk_label        text check (risk_label in ('HIGH', 'LOW')),
  contract          text    not null,
  monthly_charges   double precision not null check (monthly_charges >= 0),
  tenure_months     integer not null check (tenure_months >= 0),
  internet_service  text    not null,
  payment_method    text,
  services_count    integer not null default 0,
  high_risk_flag    integer not null default 0,
  true_label        integer,                      -- ground truth (for metrics)
  all_features      jsonb   not null,             -- pre-encoding feature dict fed to the agents
  top_shap_drivers  jsonb   not null default '[]'::jsonb,
  model_version     text
);
create index customers_risk_idx  on public.customers (risk_score desc);
create index customers_label_idx on public.customers (risk_label);

-- Every scoring run, append-only (the latest score is denormalised onto customers).
create table public.predictions (
  id             bigint generated always as identity primary key,
  customer_id    bigint not null references public.customers(customer_id) on delete cascade,
  model_version  text   not null,
  risk_score     double precision not null,
  risk_label     text   not null,
  threshold      double precision not null,
  shap_drivers   jsonb  not null default '[]'::jsonb,
  scored_at      timestamptz not null default now()
);
create index predictions_customer_idx on public.predictions (customer_id, scored_at desc);

-- Retention offers drafted by the war room (one row per run).
create table public.recommendations (
  id              bigint generated always as identity primary key,
  log_id          text not null,
  customer_id     text not null,                  -- e.g. "TEST-1234"
  risk_score      double precision not null,
  offer_text      text not null,
  contract_type   text not null,
  monthly_charge  double precision not null,
  timestamp       text not null,
  status          text not null,
  assigned_to     text not null,
  created_at      timestamptz not null default now()
);
create index recommendations_customer_idx on public.recommendations (customer_id);
create index recommendations_status_idx   on public.recommendations (status);

-- Synthetic CRM support tickets, read by the MCP server's get_open_tickets tool.
create table public.crm_tickets (
  id           bigint generated always as identity primary key,
  customer_id  bigint not null references public.customers(customer_id) on delete cascade,
  subject      text   not null,
  category     text   not null,
  status       text   not null check (status in ('Open', 'Resolved')),
  priority     text   not null check (priority in ('Low', 'Medium', 'High')),
  opened_at    text   not null
);
create index crm_tickets_customer_idx on public.crm_tickets (customer_id);

-- Small key/value store: decision threshold, model version, seed version.
create table public.meta (
  key    text primary key,
  value  text not null
);

alter table public.customers       enable row level security;
alter table public.predictions     enable row level security;
alter table public.recommendations enable row level security;
alter table public.crm_tickets     enable row level security;
alter table public.meta            enable row level security;
