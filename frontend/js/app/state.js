let currentState = null;

let connectionState = {
  kind: "boot",
  transport: "none",
  connected: false,
  reconnectAttempt: 0,
};

export function getCurrentState() {
  return currentState;
}

export function setCurrentState(nextState) {
  currentState = nextState;
}

export function getConnectionState() {
  return connectionState;
}

export function setConnectionState(nextState) {
  connectionState = {
    ...connectionState,
    ...nextState,
  };
}