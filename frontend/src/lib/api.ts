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
  WSAgent,
  WSDone,
  WSMessage,
} from "./types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

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
  const url = `${API_BASE}/api/analyze/${id}/stream?customer_state=${customerState}`;
  const es = new EventSource(url);

  es.addEventListener("agent1_complete", (e) =>
    handlers.onAgent1?.(JSON.parse((e as MessageEvent).data))
  );
  es.addEventListener("agent2_complete", (e) =>
    handlers.onAgent2?.(JSON.parse((e as MessageEvent).data))
  );
  es.addEventListener("agent3_complete", (e) =>
    handlers.onAgent3?.(JSON.parse((e as MessageEvent).data))
  );
  es.addEventListener("done", (e) => {
    handlers.onDone?.(JSON.parse((e as MessageEvent).data));
    es.close();
  });
  es.addEventListener("error", (e) => {
    // SSE 'error' fires both on our custom error event and on connection drop
    const msgEvent = e as MessageEvent;
    if (msgEvent.data) {
      try {
        handlers.onError?.(JSON.parse(msgEvent.data).message);
      } catch {
        handlers.onError?.("Stream error");
      }
    }
    es.close();
  });

  return es;
}

// ── Analysis (WebSocket, token streaming) ─────────────────────────────────────
export interface WSHandlers {
  onAgentStart?: (agent: WSAgent) => void;
  onToken?: (agent: WSAgent, text: string) => void;
  onAgentComplete?: (agent: WSAgent, data: Record<string, unknown>) => void;
  onDone?: (d: WSDone) => void;
  onError?: (msg: string) => void;
}

/**
 * Opens a WebSocket to the war-room stream (token-level).
 * Returns the WebSocket so the caller can close() it on unmount.
 */
export function streamAnalysisWS(
  id: number,
  customerState: string,
  handlers: WSHandlers
): WebSocket {
  const wsBase = API_BASE.replace(/^http/, "ws"); // http→ws, https→wss
  const ws = new WebSocket(
    `${wsBase}/api/analyze/${id}/ws?customer_state=${customerState}`
  );

  ws.onmessage = (e) => {
    let m: WSMessage;
    try {
      m = JSON.parse(e.data as string);
    } catch {
      return;
    }
    switch (m.type) {
      case "agent_start":
        handlers.onAgentStart?.(m.agent);
        break;
      case "token":
        handlers.onToken?.(m.agent, m.text);
        break;
      case "agent_complete":
        handlers.onAgentComplete?.(m.agent, m.data);
        break;
      case "done":
        handlers.onDone?.(m);
        ws.close();
        break;
      case "error":
        handlers.onError?.(m.message);
        ws.close();
        break;
    }
  };
  ws.onerror = () => handlers.onError?.("WebSocket connection error");

  return ws;
}

// ── Logs ──────────────────────────────────────────────────────────────────────
export function getLogs(limit = 20): Promise<RetentionLogEntry[]> {
  return getJSON<RetentionLogEntry[]>(`/api/logs?limit=${limit}`);
}

// ── Metrics ─────────────────────────────────────────────────────────────────────
export const getMetrics = () => getJSON<MetricsResponse>("/api/metrics");