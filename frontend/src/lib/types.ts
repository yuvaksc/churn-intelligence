// src/lib/types.ts
// Mirrors api/schemas.py — keep these in sync with the FastAPI Pydantic models.

export interface ShapDriver {
  feature: string;
  shap_value: number;
  raw_value: number;
  direction: string; // "increases churn risk" | "decreases churn risk"
}

export interface CustomerSummary {
  customer_id: number;
  risk_score: number;
  risk_label: "HIGH" | "LOW";
  contract: string;
  monthly_charges: number;
  tenure_months: number;
  internet_service: string;
  services_count: number;
  high_risk_flag: number;
  true_label: number | null;
}

export interface CustomerDetail extends CustomerSummary {
  all_features: Record<string, string | number>;
  top_shap_drivers: ShapDriver[];
}

export interface SimilarProfile {
  document: string;
  metadata: Record<string, string | number>;
  similarity: number;
}

export interface ChurnReason {
  reason: string;
  metadata: Record<string, string | number>;
  similarity: number;
}

export interface WarRoomResult {
  customer_id: string;
  risk_score: number;
  risk_label: string;
  risk_summary: string;
  shap_drivers: ShapDriver[];
  evidence_report: string;
  similar_profiles: SimilarProfile[];
  churn_reasons: ChurnReason[];
  retention_offer: string;
  policy: Record<string, unknown>;
  competitor_intel: Record<string, unknown>;
  crm_log_id: string;
  crm_logged: boolean;
}

// SSE event payloads
export interface Agent1Event {
  risk_score: number;
  risk_label: string;
  shap_drivers: ShapDriver[];
  risk_summary: string;
}

export interface Agent2Event {
  evidence_report: string;
  similar_profiles: SimilarProfile[];
  churn_reasons: ChurnReason[];
}

export interface Agent3Event {
  retention_offer: string;
  policy: Record<string, unknown>;
  competitor_intel: Record<string, unknown>;
  crm_log_id: string;
  crm_logged: boolean;
}

export interface DoneEvent {
  status: string;
  risk_label: string;
}

export interface RetentionLogEntry {
  log_id: string;
  customer_id: string;
  risk_score: number;
  offer_text: string;
  contract_type: string;
  monthly_charge: number;
  timestamp: string;
  status: string;
  assigned_to: string;
}

export interface MetricsResponse {
  threshold: number;
  f1: number;
  precision: number;
  recall: number;
  roc_auc: number;
  avg_precision: number;
  true_positives: number;
  false_positives: number;
  false_negatives: number;
  true_negatives: number;
  interventions_fired: number;
  actual_churners: number;
  revenue_at_risk_caught_monthly: number;
  revenue_missed_monthly: number;
  false_positive_spend_monthly: number;
}

export interface HealthResponse {
  status: string;
  model: string;
  threshold: number;
  test_set_size: number;
  high_risk_count: number;
  chroma_collections: string[];
}