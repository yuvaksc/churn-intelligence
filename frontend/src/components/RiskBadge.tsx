// src/components/RiskBadge.tsx
"use client";

export function RiskBadge({
  label,
  score,
  size = "md",
}: {
  label: string;
  score?: number;
  size?: "sm" | "md";
}) {
  const high = label === "HIGH";
  const color = high ? "var(--risk)" : "var(--safe)";
  const dim = high ? "var(--risk-dim)" : "var(--safe-dim)";
  const pad = size === "sm" ? "px-2 py-0.5 text-[11px]" : "px-2.5 py-1 text-xs";

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded font-mono font-medium tracking-wide ${pad}`}
      style={{ background: dim, color }}
    >
      <span
        className="inline-block rounded-full"
        style={{ width: 6, height: 6, background: color }}
      />
      {label}
      {score !== undefined && (
        <span className="tnum opacity-80">{(score * 100).toFixed(1)}%</span>
      )}
    </span>
  );
}