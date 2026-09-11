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
/**
 * A lépni következő fél királyának mezője a FEN-ből (pl. "e1"), vagy null.
 *
 * Sakkban mindig a LÉPNI KÖVETKEZŐ fél királya áll sakkban, ezért elég a
 * FEN két első mezője: a tábla és a soron lévő szín. A tábla felső sora a
 * 8-as sor, ezért megy a rank 8-tól lefelé.
 */
export function kingSquareFromFen(fen) {
  if (typeof fen !== "string") return null;

  const [placement, sideToMove] = fen.trim().split(/\s+/);
  if (!placement || !sideToMove) return null;

  const kingChar = sideToMove === "b" ? "k" : "K";
  const ranks = placement.split("/");
  if (ranks.length !== 8) return null;

  for (let i = 0; i < 8; i++) {
    let file = 0;

    for (const ch of ranks[i]) {
      if (ch >= "1" && ch <= "8") {
        file += Number(ch);
        continue;
      }
      if (ch === kingChar) {
        return `${"abcdefgh"[file]}${8 - i}`;
      }
      file += 1;
    }
  }

  return null;
}
