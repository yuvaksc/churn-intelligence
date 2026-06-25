// src/lib/api.ts
// All calls to the FastAPI backend live here.

import type {
  CustomerSummary,
  CustomerDetail,
  WarRoomResult,
  RetentionLogEntry,
  MetricsResponse,
  HealthResponse,
  Agent1Event,
  Agent2Event,
  Agent3Event,
  DoneEvent,
} from "./types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export const WARROOM_URL =
  process.env.NEXT_PUBLIC_WARROOM_URL ?? "http://localhost:8000";

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

// ── Health ──────────────────────────────────────────────────────────────────
export const getHealth = () => getJSON<HealthResponse>("/health");

// ── Customers ─────────────────────────────────────────────────────────────────
export function listCustomers(opts: {
  limit?: number;
  offset?: number;
  riskOnly?: boolean;
} = {}): Promise<CustomerSummary[]> {
  const params = new URLSearchParams({
    limit: String(opts.limit ?? 50),
    offset: String(opts.offset ?? 0),
    risk_only: String(opts.riskOnly ?? false),
  });
  return getJSON<CustomerSummary[]>(`/api/customers?${params}`);
}

export const getCustomer = (id: number) =>
  getJSON<CustomerDetail>(`/api/customers/${id}`);

// ── Analysis (POST, full result) ──────────────────────────────────────────────
export async function analyzeCustomer(
  id: number,
  customerState = "DEFAULT"
): Promise<WarRoomResult> {
  const res = await fetch(
    `${API_BASE}/api/analyze/${id}?customer_state=${customerState}`,
    { method: "POST" }
  );
  if (!res.ok) throw new Error(`Analyze failed: ${res.status}`);
  return res.json() as Promise<WarRoomResult>;
}

// ── Analysis (SSE, streaming) ─────────────────────────────────────────────────
export interface StreamHandlers {
  onAgent1?: (d: Agent1Event) => void;
  onAgent2?: (d: Agent2Event) => void;
  onAgent3?: (d: Agent3Event) => void;
  onDone?: (d: DoneEvent) => void;
  onError?: (msg: string) => void;
}

/**
 * Opens an SSE connection to the war room stream.
 * Returns the EventSource so the caller can close() it on unmount.
 */
export function streamAnalysis(
  id: number,
  customerState: string,
  handlers: StreamHandlers
): EventSource {
  const url = `${WARROOM_URL}/analyze/${id}?customer_id=${id}&customer_state=${customerState}`;

  fetch(url)
    .then((res) => res.text())
    .then((raw) => {
      let body = raw;
      // Unwrap Lambda proxy envelope if present
      try {
        const wrapped = JSON.parse(raw);
        if (wrapped && typeof wrapped === "object" && typeof wrapped.body === "string") {
          body = wrapped.body;
        }
      } catch {
        // raw is already plain SSE text
      }

      // Parse SSE "data: {...}" lines
      const lines = body.split("\n");
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith("data:")) continue;
        const jsonStr = trimmed.slice(5).trim();
        if (!jsonStr) continue;
        try {
          const parsed = JSON.parse(jsonStr);
          const eventType = parsed.event;
          const data = parsed.data ?? parsed;
          if (eventType === "agent1_complete") handlers.onAgent1?.(data);
          else if (eventType === "agent2_complete") handlers.onAgent2?.(data);
          else if (eventType === "agent3_complete") handlers.onAgent3?.(data);
          else if (eventType === "done") handlers.onDone?.(data);
          else if (eventType === "error") handlers.onError?.(data?.message ?? "Stream error");
        } catch {
          // skip unparseable line
        }
      }
    })
    .catch((err) => handlers.onError?.(String(err)));

  // Return a dummy object so the page's esRef.current?.close() still works
  return { close: () => {} } as EventSource;
}
// ── Logs ──────────────────────────────────────────────────────────────────────
export function getLogs(limit = 20): Promise<RetentionLogEntry[]> {
  return getJSON<RetentionLogEntry[]>(`/api/logs?limit=${limit}`);
}

// ── Metrics ─────────────────────────────────────────────────────────────────────
export const getMetrics = () => getJSON<MetricsResponse>("/api/metrics");