// src/app/page.tsx
"use client";

import { useEffect, useState } from "react";
import { listCustomers, getHealth } from "@/lib/api";
import type { CustomerSummary, HealthResponse } from "@/lib/types";
import { CustomerTable } from "@/components/CustomerTable";
import { ChevronLeft, ChevronRight } from "lucide-react";

const PAGE_SIZE = 50;

export default function Dashboard() {
  const [customers, setCustomers]   = useState<CustomerSummary[]>([]);
  const [health, setHealth]         = useState<HealthResponse | null>(null);
  const [riskOnly, setRiskOnly]     = useState(true);
  const [page, setPage]             = useState(1);
  const [loading, setLoading]       = useState(true);
  const [error, setError]           = useState<string | null>(null);

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch((e) => setError(String(e)));
  }, []);

  useEffect(() => {
    setLoading(true);
    const offset = (page - 1) * PAGE_SIZE;
    listCustomers({ limit: PAGE_SIZE, offset, riskOnly })
      .then((cust) => { setCustomers(cust); setError(null); })
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, [riskOnly, page]);

  function handleFilter(newRiskOnly: boolean) {
    setRiskOnly(newRiskOnly);
    setPage(1);
  }

  const total      = riskOnly ? (health?.high_risk_count ?? 0) : (health?.test_set_size ?? 0);
  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const firstItem  = total === 0 ? 0 : (page - 1) * PAGE_SIZE + 1;
  const lastItem   = Math.min(page * PAGE_SIZE, total);

  return (
    <div className="space-y-8">

      <div>
        <h1 className="font-display font-black text-3xl tracking-tight">Risk Dashboard</h1>
        <p className="mt-1 text-sm" style={{ color: "var(--text-dim)" }}>
          Customers ranked by churn probability. Click any row to open the war room.
        </p>
      </div>

      {health && (
        <div className="grid grid-cols-4 gap-4">
          <Stat label="Test Customers"  value={health.test_set_size} />
          <Stat label="High Risk"       value={health.high_risk_count} accent="var(--risk)" />
          <Stat label="Threshold"       value={`${(health.threshold * 100).toFixed(0)}%`} accent="var(--warn)" />
          <Stat label="RAG Collections" value={health.chroma_collections.length} accent="var(--info)" />
        </div>
      )}

      <div className="flex items-center gap-3">
        <FilterBtn
          active={riskOnly}
          onClick={() => handleFilter(true)}
          label="HIGH RISK ONLY"
          count={health?.high_risk_count}
          color="var(--risk)"
          dimColor="var(--risk-dim)"
        />
        <FilterBtn
          active={!riskOnly}
          onClick={() => handleFilter(false)}
          label="ALL CUSTOMERS"
          count={health?.test_set_size}
          color="var(--text)"
          dimColor="var(--surface-3)"
        />
      </div>

      {error && (
        <div className="card p-6 font-mono text-sm" style={{ color: "var(--risk)", borderColor: "var(--risk)" }}>
          {error}
          <p className="mt-2 text-[13px]" style={{ color: "var(--text-dim)" }}>
            Is the FastAPI server running on :8000?
          </p>
        </div>
      )}

      {!error && (
        <>
          <div className="flex items-center justify-between">
            <span className="font-mono text-[13px]" style={{ color: "var(--text-dim)" }}>
              {total === 0
                ? "No customers found"
                : `Showing ${firstItem}–${lastItem} of ${total.toLocaleString()} ${riskOnly ? "high-risk" : "total"} customers`}
            </span>
          </div>

          {loading
            ? <div className="card h-64 shimmer rounded-lg" />
            : <CustomerTable customers={customers} />
          }

          {!loading && totalPages > 1 && (
            <div className="flex items-center justify-between pt-2">
              <span className="font-mono text-[13px]" style={{ color: "var(--text-faint)" }}>
                Page {page} of {totalPages}
              </span>
              <div className="flex items-center gap-2">
                <PaginationBtn disabled={page === 1} onClick={() => setPage((p) => p - 1)} icon={<ChevronLeft size={16} />} label="Previous" />
                <div className="flex items-center gap-1">
                  {getPageNumbers(page, totalPages).map((p, i) =>
                    p === "..." ? (
                      <span key={`e-${i}`} className="px-2 font-mono text-[13px]" style={{ color: "var(--text-faint)" }}>…</span>
                    ) : (
                      <button
                        key={p}
                        onClick={() => setPage(p as number)}
                        className="px-3 py-1.5 rounded-md font-mono text-[13px] transition-colors"
                        style={{
                          background: p === page ? "var(--risk-dim)" : "var(--surface)",
                          color:      p === page ? "var(--risk)"     : "var(--text-dim)",
                          border:     `1px solid ${p === page ? "var(--risk)" : "var(--border)"}`,
                        }}
                      >
                        {p}
                      </button>
                    )
                  )}
                </div>
                <PaginationBtn disabled={page === totalPages} onClick={() => setPage((p) => p + 1)} icon={<ChevronRight size={16} />} label="Next" iconRight />
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function Stat({ label, value, accent = "var(--text)" }: { label: string; value: string | number; accent?: string }) {
  return (
    <div className="card p-4">
      <div className="font-mono text-[11px] uppercase tracking-wider" style={{ color: "var(--text-faint)" }}>{label}</div>
      <div className="mt-1.5 font-display font-extrabold text-2xl tnum" style={{ color: accent }}>
        {typeof value === "number" ? value.toLocaleString() : value}
      </div>
    </div>
  );
}

function FilterBtn({ active, onClick, label, count, color, dimColor }: {
  active: boolean; onClick: () => void; label: string;
  count?: number; color: string; dimColor: string;
}) {
  return (
    <button
      onClick={onClick}
      className="flex items-center gap-2 px-3 py-1.5 rounded-md text-[13px] font-medium font-mono transition-colors"
      style={{
        background: active ? dimColor : "var(--surface)",
        color:      active ? color    : "var(--text-dim)",
        border:     `1px solid ${active ? color : "var(--border)"}`,
      }}
    >
      {label}
      {count !== undefined && (
        <span className="px-1.5 py-0.5 rounded text-[11px] font-bold tnum" style={{
          background: active ? "rgba(255,255,255,0.1)" : "var(--surface-3)",
          color:      active ? color : "var(--text-faint)",
        }}>
          {count.toLocaleString()}
        </span>
      )}
    </button>
  );
}

function PaginationBtn({ disabled, onClick, icon, label, iconRight = false }: {
  disabled: boolean; onClick: () => void; icon: React.ReactNode; label: string; iconRight?: boolean;
}) {
  return (
    <button disabled={disabled} onClick={onClick}
      className="flex items-center gap-1 px-3 py-1.5 rounded-md text-[13px] font-medium transition-colors"
      style={{
        background: "var(--surface)", border: "1px solid var(--border)",
        color:   disabled ? "var(--text-faint)" : "var(--text-dim)",
        cursor:  disabled ? "not-allowed"       : "pointer",
        opacity: disabled ? 0.5 : 1,
      }}
    >
      {!iconRight && icon}{label}{iconRight && icon}
    </button>
  );
}

function getPageNumbers(current: number, total: number): (number | string)[] {
  if (total <= 7) return Array.from({ length: total }, (_, i) => i + 1);
  const pages: (number | string)[] = [];
  const add = (n: number) => { if (!pages.includes(n)) pages.push(n); };
  add(1);
  if (current > 3) pages.push("...");
  for (let p = Math.max(2, current - 1); p <= Math.min(total - 1, current + 1); p++) add(p);
  if (current < total - 2) pages.push("...");
  add(total);
  return pages;
}