/**
 * A bábukészlet EGYETLEN forrása.
 *
 * Két helyen kell: a chessboard.js `pieceTheme`-jéhez (a táblán álló bábuk)
 * és a leütött bábuk sávjához. Ha a kettő külön lenne megírva, a készlet
 * cseréjekor az egyik némán a régin maradna — ezért van itt közösen.
 *
 * A készlet cseréjéhez ELÉG ezt az egy sort átírni; az elérhető mappák:
 * wikipedia, chesscom_real, chesscom_pieces, chesscom_icy,
 * chesscom_glass_pieces.
 */
export const PIECE_THEME = "wikipedia";

const BASE = "assets/chessboardjs-1.0.0/img/chesspieces";

/** @param {string} piece  chessboard.js-kód, pl. "wQ" vagy "bp" */
export function pieceImageUrl(piece) {
  return `${BASE}/${PIECE_THEME}/${String(piece).toLowerCase()}.png`;
}
