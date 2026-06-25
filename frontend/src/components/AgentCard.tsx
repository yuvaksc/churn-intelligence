// src/components/AgentCard.tsx
"use client";

import { Check, Loader2, Circle, MinusCircle } from "lucide-react";

export type AgentStatus = "idle" | "active" | "complete" | "skipped";

export function AgentCard({
  index,
  title,
  subtitle,
  accent,
  status,
  children,
}: {
  index: number;
  title: string;
  subtitle: string;
  accent: string;
  status: AgentStatus;
  children?: React.ReactNode;
}) {
  const dim = status === "idle" || status === "skipped";

  return (
    <div
      className={`card relative overflow-hidden transition-all duration-500 ${
        status === "complete" ? "rise-in" : ""
      }`}
      style={{
        borderColor: status === "active" ? accent : "var(--border)",
        opacity: dim ? 0.55 : 1,
        boxShadow: status === "active" ? `0 0 0 1px ${accent}, 0 0 28px ${accent}22` : "none",
      }}
    >
      {/* left accent bar */}
      <div
        className="absolute left-0 top-0 bottom-0 w-1 transition-colors"
        style={{ background: status === "idle" || status === "skipped" ? "var(--border)" : accent }}
      />

      {/* header */}
      <div className="flex items-center justify-between px-5 py-4">
        <div className="flex items-center gap-3">
          <span
            className="font-mono font-bold text-xs px-2 py-1 rounded"
            style={{ background: "var(--surface-3)", color: accent }}
          >
            A{index}
          </span>
          <div>
            <div className="font-display font-bold text-[15px]">{title}</div>
            <div className="font-mono text-[11px]" style={{ color: "var(--text-faint)" }}>
              {subtitle}
            </div>
          </div>
        </div>
        <StatusIcon status={status} accent={accent} />
      </div>

      {/* body */}
      {status === "complete" && children && (
        <div className="px-5 pb-5 pt-1 border-t" style={{ borderColor: "var(--border)" }}>
          {children}
        </div>
      )}
      {status === "active" && (
        <div className="px-5 pb-5 space-y-2">
          <div className="h-3 rounded shimmer w-3/4" />
          <div className="h-3 rounded shimmer w-full" />
          <div className="h-3 rounded shimmer w-2/3" />
        </div>
      )}
      {status === "skipped" && (
        <div className="px-5 pb-4 font-mono text-[12px]" style={{ color: "var(--text-faint)" }}>
          Skipped — customer below intervention threshold
        </div>
      )}
    </div>
  );
}

function StatusIcon({ status, accent }: { status: AgentStatus; accent: string }) {
  if (status === "complete")
    return <Check size={18} style={{ color: accent }} />;
  if (status === "active")
    return <Loader2 size={18} className="animate-spin" style={{ color: accent }} />;
  if (status === "skipped")
    return <MinusCircle size={16} style={{ color: "var(--text-faint)" }} />;
  return <Circle size={16} style={{ color: "var(--text-faint)" }} />;
}