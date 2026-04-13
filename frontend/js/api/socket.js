import {
  WS_STATE_URL,
  WS_RECONNECT_DELAYS_MS,
  WS_PING_INTERVAL_MS,
} from "../app/config.js";

export class StateSocket {
  constructor({
    url = WS_STATE_URL,
    onState = () => {},
    onStatus = () => {},
  } = {}) {
    this.url = url;
    this.onState = onState;
    this.onStatus = onStatus;

    this.ws = null;
    this.closedManually = false;
    this.reconnectAttempt = 0;
    this.reconnectTimer = null;
    this.pingTimer = null;
  }

  connect() {
    this.closedManually = false;
    this._openSocket();
  }

  disconnect() {
    this.closedManually = true;
    this._clearReconnect();
    this._clearPing();

    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }

    this.onStatus({
      kind: "closed",
      transport: "websocket",
      connected: false,
      reconnectAttempt: this.reconnectAttempt,
    });
  }

  _openSocket() {
    this.onStatus({
      kind: "connecting",
      transport: "websocket",
      connected: false,
      reconnectAttempt: this.reconnectAttempt,
    });

    this.ws = new WebSocket(this.url);

    this.ws.addEventListener("open", () => {
      this.reconnectAttempt = 0;
      this.onStatus({
        kind: "open",
        transport: "websocket",
        connected: true,
        reconnectAttempt: 0,
      });
      this._startPing();
    });

    this.ws.addEventListener("message", (event) => {
      try {
        const msg = JSON.parse(event.data);

        if (msg?.type === "state") {
          this.onState(msg.payload, { via: "ws" });
          return;
        }

        if (msg?.type === "pong") {
          return;
        }
      } catch (err) {
        console.error("WebSocket üzenet parse hiba:", err);
      }
    });

    this.ws.addEventListener("error", () => {
      this.onStatus({
        kind: "error",
        transport: "websocket",
        connected: false,
        reconnectAttempt: this.reconnectAttempt,
      });
    });

    this.ws.addEventListener("close", () => {
      this._clearPing();

      if (this.closedManually) return;

      this.onStatus({
        kind: "disconnected",
        transport: "websocket",
        connected: false,
        reconnectAttempt: this.reconnectAttempt,
      });

      this._scheduleReconnect();
    });
  }

  _scheduleReconnect() {
    this._clearReconnect();

    const idx = Math.min(this.reconnectAttempt, WS_RECONNECT_DELAYS_MS.length - 1);
    const delay = WS_RECONNECT_DELAYS_MS[idx];
    this.reconnectAttempt += 1;

    this.reconnectTimer = setTimeout(() => {
      this._openSocket();
    }, delay);

    this.onStatus({
      kind: "reconnecting",
      transport: "websocket",
      connected: false,
      reconnectAttempt: this.reconnectAttempt,
    });
  }

  _startPing() {
    this._clearPing();

    this.pingTimer = setInterval(() => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send("ping");
      }
    }, WS_PING_INTERVAL_MS);
  }

  _clearReconnect() {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  _clearPing() {
    if (this.pingTimer) {
      clearInterval(this.pingTimer);
      this.pingTimer = null;
    }
  }
}