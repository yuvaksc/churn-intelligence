// src/app/metrics/page.tsx
"use client";

import { useEffect, useState } from "react";
import { getMetrics } from "@/lib/api";
import type { MetricsResponse } from "@/lib/types";
import { MetricsCard } from "@/components/MetricsCard";

export default function MetricsPage() {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getMetrics()
      .then((data) => { setMetrics(data); setError(null); })
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="space-y-8">
      <div>
        <h1 className="font-display font-black text-3xl tracking-tight">Model Metrics</h1>
        <p className="mt-1 text-sm" style={{ color: "var(--text-dim)" }}>
          XGBoost performance on the held-out test set.
        </p>
      </div>

      {loading && (
        <div className="space-y-4">
          <div className="grid grid-cols-4 gap-4">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="card h-24 shimmer" />
            ))}
          </div>
          <div className="grid grid-cols-3 gap-4">
            {[...Array(3)].map((_, i) => (
              <div key={i} className="card h-24 shimmer" />
            ))}
          </div>
        </div>
      )}

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

      {!loading && !error && metrics && (
        <div className="space-y-6">
          {/* Threshold + interventions strip */}
          <div className="grid grid-cols-2 gap-4">
            <MetricsCard
              label="Decision Threshold"
              value={`${(metrics.threshold * 100).toFixed(0)}%`}
              accent="var(--warn)"
              subtitle="Probability cutoff for HIGH risk"
            />
            <MetricsCard
              label="Interventions Fired"
              value={metrics.interventions_fired}
              accent="var(--info)"
              subtitle={`of ${metrics.actual_churners} actual churners`}
            />
          </div>

          {/* Classification metrics */}
          <div>
            <SectionLabel>Classification Performance</SectionLabel>
            <div className="grid grid-cols-4 gap-4">
              <MetricsCard
                label="F1 Score"
                value={`${(metrics.f1 * 100).toFixed(1)}%`}
                accent="var(--info)"
              />
              <MetricsCard
                label="Precision"
                value={`${(metrics.precision * 100).toFixed(1)}%`}
                accent="var(--info)"
              />
              <MetricsCard
                label="Recall"
                value={`${(metrics.recall * 100).toFixed(1)}%`}
                accent="var(--info)"
              />
              <MetricsCard
                label="ROC-AUC"
                value={`${(metrics.roc_auc * 100).toFixed(1)}%`}
                accent="var(--info)"
              />
            </div>
          </div>

          {/* Confusion matrix */}
          <div>
            <SectionLabel>Confusion Matrix</SectionLabel>
            <div className="grid grid-cols-4 gap-4">
              <MetricsCard
                label="True Positives"
                value={metrics.true_positives}
                accent="var(--safe)"
                subtitle="Correctly flagged churners"
              />
              <MetricsCard
                label="False Positives"
                value={metrics.false_positives}
                accent="var(--warn)"
                subtitle="Incorrectly flagged"
              />
              <MetricsCard
                label="False Negatives"
                value={metrics.false_negatives}
                accent="var(--risk)"
                subtitle="Missed churners"
              />
              <MetricsCard
                label="True Negatives"
                value={metrics.true_negatives}
                accent="var(--text-dim)"
                subtitle="Correctly cleared"
              />
            </div>
          </div>

          {/* Revenue impact */}
          <div>
            <SectionLabel>Monthly Revenue Impact</SectionLabel>
            <div className="grid grid-cols-3 gap-4">
              <MetricsCard
                label="Revenue Caught"
                value={`$${metrics.revenue_at_risk_caught_monthly.toFixed(0)}`}
                accent="var(--safe)"
                subtitle="At-risk MRR intercepted"
              />
              <MetricsCard
                label="Revenue Missed"
                value={`$${metrics.revenue_missed_monthly.toFixed(0)}`}
                accent="var(--risk)"
                subtitle="Churners we didn't catch"
              />
              <MetricsCard
                label="False Positive Spend"
                value={`$${metrics.false_positive_spend_monthly.toFixed(0)}`}
                accent="var(--warn)"
                subtitle="Offers sent to non-churners"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div
      className="font-mono text-[11px] uppercase tracking-wider mb-3"
      style={{ color: "var(--text-faint)" }}
    >
      {children}
    </div>
  );
}
