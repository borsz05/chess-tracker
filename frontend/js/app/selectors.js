export function selectFen(state) {
  return state?.fen ?? null;
}

export function selectMoves(state) {
  return Array.isArray(state?.moves) ? state.moves : [];
}

export function selectMovesSignature(state) {
  const moves = selectMoves(state);
  return moves.map((m) => `${m.ply_index}|${m.uci}|${m.san ?? ""}`).join("||");
}

export function selectTopLines(state) {
  return Array.isArray(state?.top_lines) ? state.top_lines : [];
}

export function selectTopLinesSignature(state) {
  // Az eredménynek is benne kell lennie: a partit lezáró lépésnél az állapot
  // már tartalmazza a "0-1"-et, de a motorvonalak még az előző pozícióhoz
  // tartoznak. Enélkül az eval-sáv és a vonalak csak akkor váltanának át a
  // győzelmi nézetre, amikor a motor is válaszol — addig egy elavult
  // centipawn-érték látszana egy már eldőlt partin.
  return JSON.stringify({
    lines: selectTopLines(state),
    result: state?.result ?? null,
  });
}

export function selectStatusSignature(state) {
  const status = state?.status ?? {};
  return JSON.stringify({
    result: state?.result ?? null,
    side_to_move: state?.side_to_move ?? null,
    fullmove_number: state?.fullmove_number ?? null,
    halfmove_clock: state?.halfmove_clock ?? null,
    analysis_pending: !!state?.analysis_pending,
    status,
  });
}

export function selectBoardSignature(state) {
  return JSON.stringify({
    fen: state?.fen ?? null,
    last_move_uci: state?.last_move?.uci ?? null,
    move_count: Array.isArray(state?.moves) ? state.moves.length : 0,
  });
}
