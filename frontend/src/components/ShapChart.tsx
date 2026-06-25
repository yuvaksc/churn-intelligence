// src/components/ShapChart.tsx
"use client";

import type { ShapDriver } from "@/lib/types";

export function ShapChart({ drivers }: { drivers: ShapDriver[] }) {
  if (!drivers.length) return null;

  const maxAbs = Math.max(...drivers.map((d) => Math.abs(d.shap_value)));

  return (
    <div className="space-y-2">
      {drivers.map((d, i) => {
        const pct = maxAbs > 0 ? (Math.abs(d.shap_value) / maxAbs) * 100 : 0;
        const increases = d.direction === "increases churn risk";
        const barColor = increases ? "var(--risk)" : "var(--safe)";

        return (
          <div key={i} className="grid gap-x-3" style={{ gridTemplateColumns: "200px 1fr 72px" }}>
            {/* feature name */}
            <div className="flex items-center min-w-0">
              <span
                className="font-mono text-[11px] truncate"
                style={{ color: "var(--text-dim)" }}
                title={d.feature}
              >
                {d.feature}
              </span>
            </div>

            {/* bar track */}
            <div
              className="flex items-center rounded overflow-hidden"
              style={{ background: "var(--surface-3)", height: 14 }}
            >
              <div
                className="h-full rounded transition-all duration-700"
                style={{ width: `${pct}%`, background: barColor, opacity: 0.85 }}
              />
            </div>

            {/* raw value + direction indicator */}
            <div className="flex items-center justify-end gap-1.5">
              <span
                className="font-mono text-[11px] tnum"
                style={{ color: "var(--text-faint)" }}
              >
                {typeof d.raw_value === "number" ? d.raw_value.toFixed(2) : d.raw_value}
              </span>
              <span className="font-mono text-[10px]" style={{ color: barColor }}>
                {increases ? "↑" : "↓"}
              </span>
            </div>
          </div>
        );
      })}
    </div>
  );
}
