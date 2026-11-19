// src/app/metrics/page.tsx
"use client";

import { useEffect, useState } from "react";
import { getMetrics } from "@/lib/api";
import type { MetricsResponse } from "@/lib/types";
import { MetricsCard } from "@/components/MetricsCard";

const usd = (n: number) => `$${Math.round(n).toLocaleString("en-US")}`;
const pct = (n: number) => `${Math.round(n * 100)}%`;

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

          {/* Modeled business impact (one-time, LTV basis) */}
          <div>
            <SectionLabel>Modeled Business Impact (one-time, LTV basis)</SectionLabel>
            <div className="grid grid-cols-3 gap-4">
              <MetricsCard
                label="Expected LTV Saved"
                value={usd(metrics.expected_ltv_saved)}
                accent="var(--safe)"
                subtitle="Lifetime value retained from caught churners"
              />
              <MetricsCard
                label="Expected LTV Missed"
                value={usd(metrics.expected_ltv_missed)}
                accent="var(--risk)"
                subtitle="Same basis, for churners we didn't catch"
              />
              <MetricsCard
                label="Campaign Cost"
                value={usd(metrics.campaign_cost)}
                accent="var(--warn)"
                subtitle="Credits + discounts on every offer sent"
              />
              <MetricsCard
                label="Net Value"
                value={usd(metrics.net_value)}
                accent="var(--safe)"
                subtitle="LTV saved minus campaign cost"
              />
              <MetricsCard
                label="Return Ratio"
                value={`${metrics.return_ratio.toFixed(1)} : 1`}
                accent="var(--info)"
                subtitle="Expected value per $1 of offers"
              />
              <MetricsCard
                label="False Positive Waste"
                value={usd(metrics.false_positive_waste)}
                accent="var(--warn)"
                subtitle="Offers sent to customers who would have stayed"
              />
            </div>
            <p
              className="mt-3 font-mono text-[11px]"
              style={{ color: "var(--text-faint)" }}
            >
              Modeled estimates, not measured revenue. Assumes{" "}
              {pct(metrics.assumptions.retention_success_rate)} save rate,{" "}
              {metrics.assumptions.ltv_months}-month LTV,{" "}
              {usd(metrics.assumptions.discount_credit_usd)} credit +{" "}
              {pct(metrics.assumptions.discount_pct)} discount for{" "}
              {metrics.assumptions.discount_months} months.
            </p>
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