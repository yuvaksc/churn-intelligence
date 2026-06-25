// src/components/CustomerTable.tsx
"use client";

import Link from "next/link";
import { ArrowRight } from "lucide-react";
import type { CustomerSummary } from "@/lib/types";
import { RiskBadge } from "./RiskBadge";

export function CustomerTable({ customers }: { customers: CustomerSummary[] }) {
  return (
    <div className="card overflow-hidden">
      <table className="w-full border-collapse">
        <thead>
          <tr
            className="text-left font-mono text-[11px] uppercase tracking-wider"
            style={{ color: "var(--text-faint)" }} 
          >
            <th className="px-4 py-3 font-medium">ID</th>
            <th className="px-4 py-3 font-medium">Risk</th>
            <th className="px-4 py-3 font-medium">Contract</th>
            <th className="px-4 py-3 font-medium">Internet</th>
            <th className="px-4 py-3 font-medium text-right">Tenure</th>
            <th className="px-4 py-3 font-medium text-right">Monthly</th>
            <th className="px-4 py-3 font-medium text-right">Services</th>
            <th className="px-4 py-3 font-medium text-center">Actual</th>
            <th className="px-4 py-3" />
          </tr>
        </thead>
        <tbody>
          {customers.map((c) => (
            <tr
              key={c.customer_id}
              className="border-t transition-colors group hover:bg-[var(--surface-2)]"
              style={{ borderColor: "var(--border)" }}
            >
              <td className="px-4 py-3 font-mono text-sm" style={{ color: "var(--text-dim)" }}>
                {c.customer_id}
              </td>
              <td className="px-4 py-3">
                <RiskBadge label={c.risk_label} score={c.risk_score} size="sm" />
              </td>
              <td className="px-4 py-3 text-sm">{c.contract}</td>
              <td className="px-4 py-3 text-sm" style={{ color: "var(--text-dim)" }}>
                {c.internet_service}
              </td>
              <td className="px-4 py-3 text-right font-mono text-sm tnum">
                {c.tenure_months}mo
              </td>
              <td className="px-4 py-3 text-right font-mono text-sm tnum">
                ${c.monthly_charges.toFixed(2)}
              </td>
              <td className="px-4 py-3 text-right font-mono text-sm tnum" style={{ color: "var(--text-dim)" }}>
                {c.services_count}
              </td>
              <td className="px-4 py-3 text-center">
                {c.true_label !== null && (
                  <span
                    className="font-mono text-[11px]"
                    style={{ color: c.true_label === 1 ? "var(--risk)" : "var(--safe)" }}
                  >
                    {c.true_label === 1 ? "churned" : "stayed"}
                  </span>
                )}
              </td>
              <td className="px-4 py-3 text-right">
                <Link
                  href={`/customer/${c.customer_id}`}
                  className="inline-flex items-center gap-1 text-[13px] font-medium no-underline opacity-0 group-hover:opacity-100 transition-opacity"
                  style={{ color: "var(--accent)" }}
                >
                  War Room <ArrowRight size={14} />
                </Link>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}