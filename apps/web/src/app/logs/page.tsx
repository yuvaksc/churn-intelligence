// src/app/logs/page.tsx
"use client";

import { useEffect, useState } from "react";
import { getLogs } from "@/lib/api";
import type { RetentionLogEntry } from "@/lib/types";

export default function LogsPage() {
  const [logs, setLogs] = useState<RetentionLogEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getLogs(50)
      .then((data) => { setLogs(data); setError(null); })
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="space-y-8">
      <div>
        <h1 className="font-display font-black text-3xl tracking-tight">Retention Log</h1>
        <p className="mt-1 text-sm" style={{ color: "var(--text-dim)" }}>
          All interventions fired by the war room agents.
        </p>
      </div>

      {loading && <div className="card h-64 shimmer rounded-lg" />}

      {error && (
        <div
          className="card p-6 font-mono text-sm"
          style={{ color: "var(--risk)", borderColor: "var(--risk)" }}
        >
          {error}
          <p className="mt-2 text-[13px]" style={{ color: "var(--text-dim)" }}>
            Is the FastAPI server running on :8000?
          </p>
        </div>
      )}

      {!loading && !error && logs.length === 0 && (
        <div className="card p-8 text-center font-mono text-sm" style={{ color: "var(--text-faint)" }}>
          No interventions logged yet. Run a war room analysis to generate entries.
        </div>
      )}

      {!loading && !error && logs.length > 0 && (
        <div className="card overflow-hidden">
          <table className="w-full border-collapse">
            <thead>
              <tr
                className="text-left font-mono text-[11px] uppercase tracking-wider"
                style={{ color: "var(--text-faint)" }}
              >
                <th className="px-4 py-3 font-medium">Log ID</th>
                <th className="px-4 py-3 font-medium">Customer</th>
                <th className="px-4 py-3 font-medium text-right">Risk</th>
                <th className="px-4 py-3 font-medium">Contract</th>
                <th className="px-4 py-3 font-medium text-right">Monthly</th>
                <th className="px-4 py-3 font-medium">Offer</th>
                <th className="px-4 py-3 font-medium">Assigned To</th>
                <th className="px-4 py-3 font-medium text-center">Status</th>
                <th className="px-4 py-3 font-medium text-right">Timestamp</th>
              </tr>
            </thead>
            <tbody>
              {logs.map((log) => (
                <tr
                  key={log.log_id}
                  className="border-t transition-colors hover:bg-[var(--surface-2)]"
                  style={{ borderColor: "var(--border)" }}
                >
                  <td className="px-4 py-3 font-mono text-[11px]" style={{ color: "var(--text-faint)" }}>
                    {log.log_id.slice(0, 8)}
                  </td>
                  <td className="px-4 py-3 font-mono text-sm" style={{ color: "var(--text-dim)" }}>
                    {log.customer_id}
                  </td>
                  <td className="px-4 py-3 text-right font-mono text-sm tnum" style={{ color: "var(--risk)" }}>
                    {(log.risk_score * 100).toFixed(1)}%
                  </td>
                  <td className="px-4 py-3 text-sm">{log.contract_type}</td>
                  <td className="px-4 py-3 text-right font-mono text-sm tnum">
                    ${log.monthly_charge.toFixed(2)}
                  </td>
                  <td
                    className="px-4 py-3 text-sm max-w-[240px] truncate"
                    style={{ color: "var(--text-dim)" }}
                    title={log.offer_text}
                  >
                    {log.offer_text}
                  </td>
                  <td className="px-4 py-3 font-mono text-[12px]" style={{ color: "var(--text-faint)" }}>
                    {log.assigned_to}
                  </td>
                  <td className="px-4 py-3 text-center">
                    <span
                      className="font-mono text-[11px]"
                      style={{ color: log.status === "logged" ? "var(--safe)" : "var(--text-faint)" }}
                    >
                      {log.status}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-right font-mono text-[11px]" style={{ color: "var(--text-faint)" }}>
                    {new Date(log.timestamp).toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
