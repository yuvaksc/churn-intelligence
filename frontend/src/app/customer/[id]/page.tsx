// src/app/customer/[id]/page.tsx
"use client";

import { use, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { ArrowLeft, CheckCircle2 } from "lucide-react";
import { getCustomer, streamAnalysisWS } from "@/lib/api";
import type {
  CustomerDetail,
  Agent1Event,
  Agent2Event,
  Agent3Event,
} from "@/lib/types";
import { AgentCard } from "@/components/AgentCard";
import type { AgentStatus } from "@/components/AgentCard";
import { RiskBadge } from "@/components/RiskBadge";
import { ShapChart } from "@/components/ShapChart";

const CUSTOMER_STATES = ["DEFAULT", "LOYAL", "NEW", "ENTERPRISE"] as const;

export default function WarRoomPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const numericId = parseInt(id, 10);

  const [customer, setCustomer] = useState<CustomerDetail | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [customerState, setCustomerState] = useState<string>("DEFAULT");

  const [a1Status, setA1Status] = useState<AgentStatus>("idle");
  const [a2Status, setA2Status] = useState<AgentStatus>("idle");
  const [a3Status, setA3Status] = useState<AgentStatus>("idle");

  const [a1Data, setA1Data] = useState<Agent1Event | null>(null);
  const [a2Data, setA2Data] = useState<Agent2Event | null>(null);
  const [a3Data, setA3Data] = useState<Agent3Event | null>(null);

  // live token buffers (shown while an agent is "active")
  const [a1Tok, setA1Tok] = useState("");
  const [a2Tok, setA2Tok] = useState("");
  const [a3Tok, setA3Tok] = useState("");

  const [running, setRunning] = useState(false);
  const [done, setDone] = useState(false);
  const [streamError, setStreamError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    getCustomer(numericId)
      .then((c) => { setCustomer(c); setLoadError(null); })
      .catch((e) => setLoadError(String(e)));
  }, [numericId]);

  useEffect(() => {
    return () => { wsRef.current?.close(); };
  }, []);

  function resetAgents() {
    setA1Status("idle"); setA1Data(null); setA1Tok("");
    setA2Status("idle"); setA2Data(null); setA2Tok("");
    setA3Status("idle"); setA3Data(null); setA3Tok("");
    setDone(false);
    setStreamError(null);
  }

  function runAnalysis() {
    wsRef.current?.close();
    resetAgents();
    setRunning(true);
    setA1Status("active");   // optimistic: the supervisor routes to agent1 first

    wsRef.current = streamAnalysisWS(numericId, customerState, {
      onAgentStart(agent) {
        if (agent === "agent1") setA1Status("active");
        if (agent === "agent2") setA2Status("active");
        if (agent === "agent3") setA3Status("active");
      },
      onToken(agent, text) {
        if (agent === "agent1") setA1Tok((s) => s + text);
        if (agent === "agent2") setA2Tok((s) => s + text);
        if (agent === "agent3") setA3Tok((s) => s + text);
      },
      onAgentComplete(agent, data) {
        if (agent === "agent1") {
          const d = data as unknown as Agent1Event;
          setA1Data(d);
          setA1Status("complete");
          // keep the pipeline visibly working through the supervisor's routing +
          // agent2's retrieval, so there's no "dead" gap where nothing spins
          if (d.risk_label === "HIGH") setA2Status("active");
        }
        if (agent === "agent2") {
          setA2Data(data as unknown as Agent2Event);
          setA2Status("complete");
          setA3Status("active");
        }
        if (agent === "agent3") {
          setA3Data(data as unknown as Agent3Event);
          setA3Status("complete");
        }
      },
      onDone() {
        setDone(true);
        setRunning(false);
        // anything that didn't actually complete (low-risk skip, or an optimistic
        // activation that never ran) → skipped, so nothing is left spinning
        setA2Status((s) => (s === "complete" ? s : "skipped"));
        setA3Status((s) => (s === "complete" ? s : "skipped"));
      },
      onError(msg) {
        setStreamError(msg);
        setRunning(false);
        setA1Status((s) => (s === "active" ? "idle" : s));
        setA2Status((s) => (s === "active" ? "idle" : s));
        setA3Status((s) => (s === "active" ? "idle" : s));
      },
    });
  }

  if (loadError) {
    return (
      <div className="space-y-4">
        <BackLink />
        <div
          className="card p-6 font-mono text-sm"
          style={{ color: "var(--risk)", borderColor: "var(--risk)" }}
        >
          {loadError}
        </div>
      </div>
    );
  }

  if (!customer) {
    return (
      <div className="space-y-4">
        <BackLink />
        <div className="card h-32 shimmer" />
        <div className="card h-48 shimmer" />
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <BackLink />

      {/* Customer header */}
      <div className="card p-6 space-y-5">
        <div className="flex items-start justify-between">
          <div>
            <div className="font-mono text-[11px] uppercase tracking-wider" style={{ color: "var(--text-faint)" }}>
              Customer ID
            </div>
            <div className="font-display font-extrabold text-3xl mt-0.5">{customer.customer_id}</div>
          </div>
          <RiskBadge label={customer.risk_label} score={customer.risk_score} size="md" />
        </div>

        <div className="grid grid-cols-4 gap-4">
          <DetailStat label="Contract" value={customer.contract} />
          <DetailStat label="Internet" value={customer.internet_service} />
          <DetailStat label="Tenure" value={`${customer.tenure_months} mo`} />
          <DetailStat label="Monthly" value={`$${customer.monthly_charges.toFixed(2)}`} />
        </div>

        {customer.true_label !== null && (
          <div className="font-mono text-[12px]" style={{ color: "var(--text-faint)" }}>
            Ground truth:{" "}
            <span style={{ color: customer.true_label === 1 ? "var(--risk)" : "var(--safe)" }}>
              {customer.true_label === 1 ? "churned" : "stayed"}
            </span>
          </div>
        )}
      </div>

      {/* SHAP drivers */}
      {customer.top_shap_drivers.length > 0 && (
        <div className="card p-6">
          <div className="font-mono text-[11px] uppercase tracking-wider mb-4" style={{ color: "var(--text-faint)" }}>
            Top SHAP Drivers
          </div>
          <ShapChart drivers={customer.top_shap_drivers} />
        </div>
      )}

      {/* Analysis controls */}
      <div className="flex items-center gap-4">
        <select
          value={customerState}
          onChange={(e) => setCustomerState(e.target.value)}
          disabled={running}
          className="font-mono text-[13px] rounded-md px-3 py-1.5"
          style={{
            background: "var(--surface)",
            border: "1px solid var(--border)",
            color: "var(--text-dim)",
          }}
        >
          {CUSTOMER_STATES.map((s) => (
            <option key={s} value={s}>{s}</option>
          ))}
        </select>

        <button
          onClick={runAnalysis}
          disabled={running}
          className="px-4 py-1.5 rounded-md text-[13px] font-medium font-mono transition-colors"
          style={{
            background: running ? "var(--surface-3)" : "var(--risk-dim)",
            color: running ? "var(--text-faint)" : "var(--risk)",
            border: `1px solid ${running ? "var(--border)" : "var(--risk)"}`,
            cursor: running ? "not-allowed" : "pointer",
          }}
        >
          {running ? "Streaming…" : done ? "Re-run Analysis" : "Run Analysis"}
        </button>

        {done && (
          <span className="flex items-center gap-1.5 font-mono text-[12px]" style={{ color: "var(--safe)" }}>
            <CheckCircle2 size={14} /> Analysis complete
          </span>
        )}
      </div>

      {streamError && (
        <div
          className="card p-4 font-mono text-sm"
          style={{ color: "var(--risk)", borderColor: "var(--risk)" }}
        >
          {streamError}
        </div>
      )}

      {/* Agent cards */}
      <div className="space-y-4">
        <AgentCard
          index={1}
          title="Risk Analyst"
          subtitle="XGBoost scoring · SHAP attribution"
          accent="var(--risk)"
          status={a1Status}
          streamingText={a1Tok}
        >
          {a1Data && (
            <div className="space-y-4">
              <p className="text-sm leading-relaxed" style={{ color: "var(--text-dim)" }}>
                {a1Data.risk_summary}
              </p>
              {a1Data.shap_drivers.length > 0 && (
                <div>
                  <div className="font-mono text-[11px] uppercase tracking-wider mb-3" style={{ color: "var(--text-faint)" }}>
                    Agent SHAP Drivers
                  </div>
                  <ShapChart drivers={a1Data.shap_drivers} />
                </div>
              )}
            </div>
          )}
        </AgentCard>

        <AgentCard
          index={2}
          title="Evidence Collector"
          subtitle="RAG search · similar profiles · churn reasons"
          accent="var(--warn)"
          status={a2Status}
          streamingText={a2Tok}
        >
          {a2Data && (
            <div className="space-y-4">
              <p className="text-sm leading-relaxed whitespace-pre-wrap" style={{ color: "var(--text-dim)" }}>
                {a2Data.evidence_report}
              </p>
              <div className="flex gap-6 font-mono text-[12px]" style={{ color: "var(--text-faint)" }}>
                <span>{a2Data.similar_profiles.length} similar profiles</span>
                <span>{a2Data.churn_reasons.length} churn reasons</span>
              </div>
            </div>
          )}
        </AgentCard>

        <AgentCard
          index={3}
          title="Retention Strategist"
          subtitle="ReAct · dynamic MCP tools · offer"
          accent="var(--safe)"
          status={a3Status}
          streamingText={a3Tok}
        >
          {a3Data && (
            <div className="space-y-4">
              <div
                className="rounded-lg p-4 text-sm leading-relaxed"
                style={{ background: "var(--safe-dim)", color: "var(--text)" }}
              >
                {a3Data.retention_offer}
              </div>
              <div className="flex items-center gap-3 font-mono text-[11px]" style={{ color: "var(--text-faint)" }}>
                <span
                  className="flex items-center gap-1"
                  style={{ color: a3Data.crm_logged ? "var(--safe)" : "var(--risk)" }}
                >
                  <span
                    className="inline-block w-1.5 h-1.5 rounded-full"
                    style={{ background: a3Data.crm_logged ? "var(--safe)" : "var(--risk)" }}
                  />
                  {a3Data.crm_logged ? "CRM logged" : "CRM log failed"}
                </span>
                {a3Data.crm_log_id && (
                  <span>Log ID: {a3Data.crm_log_id.slice(0, 8)}</span>
                )}
              </div>
              {a3Data.tools_used && a3Data.tools_used.length > 0 && (
                <div className="flex flex-wrap items-center gap-1.5">
                  <span className="font-mono text-[11px]" style={{ color: "var(--text-faint)" }}>
                    Tools called:
                  </span>
                  {a3Data.tools_used.map((t, i) => (
                    <span
                      key={`${t}-${i}`}
                      className="font-mono text-[10px] px-1.5 py-0.5 rounded"
                      style={{ background: "var(--surface-3)", color: "var(--text-dim)" }}
                    >
                      {t}
                    </span>
                  ))}
                </div>
              )}
              {Object.keys(a3Data.policy).length > 0 && (
                <details className="group">
                  <summary
                    className="font-mono text-[11px] cursor-pointer select-none"
                    style={{ color: "var(--text-faint)" }}
                  >
                    Policy details ▸
                  </summary>
                  <pre
                    className="mt-2 text-[11px] font-mono p-3 rounded overflow-x-auto"
                    style={{ background: "var(--surface-3)", color: "var(--text-dim)" }}
                  >
                    {JSON.stringify(a3Data.policy, null, 2)}
                  </pre>
                </details>
              )}
            </div>
          )}
        </AgentCard>
      </div>
    </div>
  );
}

function BackLink() {
  return (
    <Link
      href="/"
      className="inline-flex items-center gap-1.5 font-mono text-[13px] no-underline transition-colors"
      style={{ color: "var(--text-faint)" }}
    >
      <ArrowLeft size={14} /> Dashboard
    </Link>
  );
}

function DetailStat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="font-mono text-[11px] uppercase tracking-wider" style={{ color: "var(--text-faint)" }}>
        {label}
      </div>
      <div className="mt-0.5 font-mono text-sm tnum" style={{ color: "var(--text-dim)" }}>
        {value}
      </div>
    </div>
  );
}
