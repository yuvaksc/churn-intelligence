// src/components/MetricsCard.tsx
"use client";

export function MetricsCard({
  label,
  value,
  accent = "var(--text)",
  subtitle,
}: {
  label: string;
  value: string | number;
  accent?: string;
  subtitle?: string;
}) {
  return (
    <div className="card p-4">
      <div
        className="font-mono text-[11px] uppercase tracking-wider"
        style={{ color: "var(--text-faint)" }}
      >
        {label}
      </div>
      <div
        className="mt-1.5 font-display font-extrabold text-2xl tnum"
        style={{ color: accent }}
      >
        {value}
      </div>
      {subtitle && (
        <div className="mt-1 font-mono text-[11px]" style={{ color: "var(--text-faint)" }}>
          {subtitle}
        </div>
      )}
    </div>
  );
}
