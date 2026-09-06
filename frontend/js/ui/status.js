import { getEl, setText } from "../utils/dom.js";
import {
  formatFinishReason,
  formatResultLabel,
  isGameFinished,
  sideToMoveLabel,
} from "../utils/game.js";

/**
 * Csak akkor mond valamit, ha BAJ van. Élő kapcsolatnál üres — a működő
 * alapállapotot nem kell feliratozni, a hibát viszont látni kell.
 */
function connectionText(connectionState) {
  const conn = connectionState || {};

  if (conn.connected) return "";
  if (conn.kind === "connecting") return "Kapcsolódás…";
  if (conn.kind === "reconnecting") {
    return `Újracsatlakozás… (${conn.reconnectAttempt ?? 0})`;
  }
  if (conn.kind === "error") return "Kapcsolati hiba";
  if (conn.kind === "disconnected") return "A kapcsolat megszakadt";
  if (conn.kind === "closed") return "A kapcsolat lezárult";
  return "";
}

export function updateStatusUI(state, connectionState) {
  const statusSection = document.querySelector(".status-section");
  const connectionDiv = getEl("connection-status");
  const statusDiv = getEl("game-status");
  const sideDiv = getEl("side-to-move");
  const countersDiv = getEl("move-counters");

  const safeState = state || {};
  const status = safeState.status || {};
  const finished = isGameFinished(safeState);

  const connText = connectionText(connectionState);
  setText(connectionDiv, connText);
  if (connectionDiv) {
    connectionDiv.hidden = connText === "";
  }

  let text = "Folyamatban";
  if (finished) {
    text = `${formatResultLabel(safeState.result)} – ${formatFinishReason(status)}`;
  } else if (status.is_check) {
    text = "Sakk!";
  } else if (safeState.analysis_pending) {
    text = "Elemzés folyamatban...";
  }

  setText(statusDiv, text);
  setText(sideDiv, `Következik: ${sideToMoveLabel(safeState.side_to_move)}`);
  setText(
    countersDiv,
    `Teljes lépés: ${safeState.fullmove_number ?? "-"}, fél-lépés számláló: ${safeState.halfmove_clock ?? "-"}`
  );

  if (statusSection) {
    statusSection.classList.toggle("game-over", finished);
  }
}