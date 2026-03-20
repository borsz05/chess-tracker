import { API_BASE, REQUEST_TIMEOUT_MS } from "../app/config.js";

async function requestJson(path, options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const res = await fetch(`${API_BASE}${path}`, {
      ...options,
      signal: controller.signal,
      headers: {
        "Content-Type": "application/json",
        ...(options.headers || {}),
      },
    });

    const data = await res.json().catch(() => ({}));

    if (!res.ok) {
      const detail = data?.detail || `HTTP ${res.status}`;
      throw new Error(detail);
    }

    return data;
  } finally {
    clearTimeout(timeout);
  }
}

export function fetchState() {
  return requestJson("/state", { method: "GET" });
}

export function postMove(uci) {
  return requestJson("/move", {
    method: "POST",
    body: JSON.stringify({ uci }),
  });
}

export function postNewGame() {
  return requestJson("/new-game", {
    method: "POST",
  });
}