import { getEl, setText } from "../utils/dom.js";
import {
  formatFinishReason,
  isGameFinished,
  resultBadgeText,
  sideToMoveLabel,
  winnerHeadline,
  decisiveResult,
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

/**
 * A jegyzet csak akkor mond valamit, ha BAJ van — jelenleg: sakk.
 *
 * A futó elemzés SZÁNDÉKOSAN nincs kijelezve. A motorvonalak a panel
 * tetején végig ott vannak, csak az értékük frissül, tehát az „Elemzés…"
 * nem mondott semmit, amit ne lehetne látni — viszont minden egyes lépés
 * után megjelent ~2 másodpercre, és ez okozta a státuszsáv ugrálását.
 */
function statusNote(state, finished) {
  const status = state?.status || {};

  // Lezárt partinál a panel tetején álló sáv mindent elmond (eredmény +
  // ok), a sorban nem ismételjük meg.
  if (finished) return null;

  if (status.is_check) {
    return { text: "Sakk!", kind: "note-alert" };
  }
  return null;
}

/** Teljes szélességű sáv a panel tetején — a parti vége nem kis doboz. */
function updateGameOverBanner(state, finished) {
  const banner = getEl("game-over-banner");
  if (!banner) return;

  if (!finished) {
    banner.hidden = true;
    return;
  }

  const result = decisiveResult(state);
  const title = banner.querySelector(".game-over-banner-title");
  const resultEl = banner.querySelector(".game-over-banner-result");
  const reasonEl = banner.querySelector(".game-over-banner-reason");

  setText(title, winnerHeadline(result) || "Parti vége");
  setText(resultEl, result ? resultBadgeText(result) : "");
  setText(reasonEl, formatFinishReason(state?.status || {}));

  banner.hidden = false;
}

export function updateStatusUI(state, connectionState) {
  const statusSection = document.querySelector(".status-section");
  const connectionDiv = getEl("connection-status");
  const noteDiv = getEl("game-status");
  const moveNoDiv = getEl("status-move-no");
  const sideDiv = getEl("status-side");

  const safeState = state || {};
  const finished = isGameFinished(safeState);

  const connText = connectionText(connectionState);
  setText(connectionDiv, connText);
  if (connectionDiv) {
    connectionDiv.hidden = connText === "";
  }

  setText(moveNoDiv, `${safeState.fullmove_number ?? "-"}. lépés`);
  setText(
    sideDiv,
    finished ? "Parti vége" : `${sideToMoveLabel(safeState.side_to_move)} következik`
  );

  const note = statusNote(safeState, finished);
  if (noteDiv) {
    if (note) {
      setText(noteDiv, note.text);
      noteDiv.classList.remove("note-alert", "note-info", "note-error");
      noteDiv.classList.add(note.kind);
      noteDiv.hidden = false;
    } else {
      setText(noteDiv, "");
      noteDiv.hidden = true;
    }
  }

  updateGameOverBanner(safeState, finished);

  if (statusSection) {
    statusSection.classList.toggle("game-over", finished);
  }
}
