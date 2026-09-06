export function isGameFinished(state) {
  if (!state) return false;

  if (state.result && state.result !== "*") {
    return true;
  }

  const status = state.status || {};
  return (
    status.is_checkmate ||
    status.is_stalemate ||
    status.is_fifty_move_draw ||
    status.is_threefold_repetition ||
    status.is_insufficient_material
  );
}

export function sideToMoveLabel(side) {
  return side === "w" ? "Fehér" : "Fekete";
}

export function formatResultLabel(result) {
  if (result === "1-0") return "Fehér nyert (1-0)";
  if (result === "0-1") return "Fekete nyert (0-1)";
  if (result === "1/2-1/2") return "Döntetlen (1/2-1/2)";
  return "Parti vége";
}

/** "1-0" | "0-1" | "1/2-1/2" — a lezárt parti eredménye, különben null. */
export function decisiveResult(state) {
  const r = state?.result;
  if (r === "1-0" || r === "0-1" || r === "1/2-1/2") return r;
  return null;
}

/** Rövid, ünneplő felirat az eredmény mellé. */
export function winnerHeadline(result) {
  if (result === "1-0") return "Fehér nyert!";
  if (result === "0-1") return "Fekete nyert!";
  if (result === "1/2-1/2") return "Döntetlen!";
  return "";
}

/** Az eval-sávban és a jelvényen megjelenő rövid alak. */
export function resultBadgeText(result) {
  return result === "1/2-1/2" ? "½–½" : result;
}

export function formatFinishReason(status = {}) {
  if (status.is_checkmate) return "Matt";
  if (status.is_stalemate) return "Patt";
  if (status.is_fifty_move_draw) return "50 lépés szabály";
  if (status.is_threefold_repetition) return "Háromszori ismétlés";
  if (status.is_insufficient_material) return "Anyaghiány";
  if (status.is_check) return "Sakk";
  return "Lezárva";
}