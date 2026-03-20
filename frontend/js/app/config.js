const runtimeConfig = window.__APP_CONFIG__ ?? {};

function trimTrailingSlash(value) {
  return String(value || "").replace(/\/+$/, "");
}

function toWsOrigin(httpOrigin) {
  return httpOrigin.replace(/^http/i, (m) => (m.toLowerCase() === "https" ? "wss" : "ws"));
}

export const BACKEND_ORIGIN = trimTrailingSlash(
  runtimeConfig.backendOrigin || "http://localhost:8001"
);

export const API_BASE = `${BACKEND_ORIGIN}/api`;
export const WS_STATE_URL = `${toWsOrigin(BACKEND_ORIGIN)}/api/ws/state`;

export const REQUEST_TIMEOUT_MS = 8000;
export const WS_RECONNECT_DELAYS_MS = [1000, 2000, 3000, 5000, 5000];
export const WS_PING_INTERVAL_MS = 20000;