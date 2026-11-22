// src/lib/api.ts
// All calls to the FastAPI backend live here.

import type {
  CustomerSummary,
  CustomerDetail,
  RetentionLogEntry,
  MetricsResponse,
  HealthResponse,
  WSAgent,
  WSDone,
  WSMessage,
} from "./types";

// Locally the API runs on its own port. On Vercel the web app and the API share one
// origin, so requests are same-origin (""). NEXT_PUBLIC_VERCEL_ENV is set automatically
// on every Vercel build; NEXT_PUBLIC_API_BASE overrides both.
export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ??
  (process.env.NEXT_PUBLIC_VERCEL_ENV ? "" : "http://localhost:8000");

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

// ── Analysis (WebSocket, token streaming) ─────────────────────────────────────
export interface WSHandlers {
  onAgentStart?: (agent: WSAgent) => void;
  onToken?: (agent: WSAgent, text: string) => void;
  onAgentComplete?: (agent: WSAgent, data: Record<string, unknown>) => void;
  onDone?: (d: WSDone) => void;
  onError?: (msg: string) => void;
}

export interface StreamHandle {
  close: () => void;
}

const MAX_CONNECT_ATTEMPTS = 3;

function wsBase(): string {
  if (API_BASE) return API_BASE.replace(/^http/, "ws"); // http→ws, https→wss
  const { protocol, host } = window.location;           // same-origin deployment
  return `${protocol === "https:" ? "wss" : "ws"}://${host}`;
}

/**
 * Opens a WebSocket to the war-room stream (token-level) and returns a handle the
 * caller can close() on unmount.
 *
 * Serverless WebSockets (Vercel Functions) can drop on a cold start or hit the
 * function's max duration, so a connection that fails BEFORE any frame arrives is
 * retried with backoff. A drop mid-stream is reported, not retried: re-opening would
 * re-run the analysis and draft a second offer.
 */
export function streamAnalysisWS(
  id: number,
  customerState: string,
  handlers: WSHandlers
): StreamHandle {
  let ws: WebSocket | null = null;
  let closedByCaller = false;
  let finished = false;
  let attempt = 0;

  const attach = (sock: WebSocket, onFrame: () => void) => {
    sock.onmessage = (e) => {
      onFrame();
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
          finished = true;
          handlers.onDone?.(m);
          sock.close();
          break;
        case "error":
          finished = true;
          handlers.onError?.(m.message);
          sock.close();
          break;
      }
    };
  };

  const open = () => {
    attempt += 1;
    if (closedByCaller) return;
    const params = new URLSearchParams({ customer_state: customerState });
    ws = new WebSocket(`${wsBase()}/api/analyze/${id}/ws?${params}`);
    let gotFrame = false;
    attach(ws, () => { gotFrame = true; });
    ws.onclose = () => {
      if (closedByCaller || finished) return;
      if (!gotFrame && attempt < MAX_CONNECT_ATTEMPTS) {
        setTimeout(open, 500 * 2 ** (attempt - 1)); // 0.5s, 1s
      } else {
        handlers.onError?.(gotFrame ? "Connection lost during analysis" : "WebSocket connection error");
      }
    };
  };

  open();
  return {
    close: () => {
      closedByCaller = true;
      ws?.close();
    },
  };
}

// ── Logs ──────────────────────────────────────────────────────────────────────
export function getLogs(limit = 20): Promise<RetentionLogEntry[]> {
  return getJSON<RetentionLogEntry[]>(`/api/logs?limit=${limit}`);
}

// ── Metrics ─────────────────────────────────────────────────────────────────────
export const getMetrics = () => getJSON<MetricsResponse>("/api/metrics");
