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
  return JSON.stringify(selectTopLines(state));
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
