/**
 * Leütött bábuk és anyagi előny a FEN-ből.
 *
 * Miért a FEN-ből, és nem a lépéstörténetből? Mert így a PROMÓCIÓ két
 * kényes esete magától helyes lesz, külön ág nélkül:
 *
 *  - Ha a világos gyalogja vezérré promotál, a világos gyalogszáma eggyel
 *    csökken, tehát a SÖTÉT listájában megjelenik egy világos gyalog — úgy,
 *    mintha leütötte volna. (Ez a kívánt viselkedés.)
 *  - Ha valaki leüt egy promócióból származó bábut, az nem kerül be ikonként:
 *    a promotált vezérből a félnek KETTŐ volt, leütés után EGY, vagyis a
 *    kezdőkészlethez képest nincs hiány, így nincs mit kirajzolni.
 *
 * Az ELŐNY viszont nem a leütöttek összegének a különbsége, hanem a táblán
 * lévő anyag különbsége. Promóciónál a kettő eltér: a fenti példában a
 * leütött-összeg alapján a világos −5 lenne, valójában viszont +12.
 */

export const PIECE_ORDER = ["p", "b", "n", "r", "q"];

export const PIECE_VALUE = { p: 1, n: 3, b: 3, r: 5, q: 9 };

const START_COUNT = { p: 8, n: 2, b: 2, r: 2, q: 1 };

/** A FEN tábla-részéből: { w: {p: 8, n: 2, ...}, b: {...} } */
function countPieces(placement) {
  const counts = {
    w: { p: 0, n: 0, b: 0, r: 0, q: 0 },
    b: { p: 0, n: 0, b: 0, r: 0, q: 0 },
  };

  for (const ch of placement) {
    if (ch === "/" || (ch >= "1" && ch <= "8")) continue;

    const lower = ch.toLowerCase();
    if (!(lower in PIECE_VALUE)) continue; // a király nem számít anyagban

    counts[ch === lower ? "b" : "w"][lower] += 1;
  }

  return counts;
}

/**
 * @returns {{
 *   capturedByWhite: Array<{type: string, count: number}>,
 *   capturedByBlack: Array<{type: string, count: number}>,
 *   advantage: number,   // > 0: a világos vezet, < 0: a sötét
 * }}
 * A listák a megjelenítés sorrendjében: gyalog, futó, huszár, bástya, vezér.
 */
export function materialFromFen(fen) {
  const empty = { capturedByWhite: [], capturedByBlack: [], advantage: 0 };
  if (typeof fen !== "string") return empty;

  const placement = fen.trim().split(/\s+/)[0];
  if (!placement) return empty;

  const counts = countPieces(placement);

  // Amit az EGYIK fél elvesztett, az a MÁSIK fél listájában jelenik meg.
  const missing = (side) =>
    PIECE_ORDER
      .map((type) => ({ type, count: START_COUNT[type] - counts[side][type] }))
      // A negatív (promócióból származó többlet) nem hiány: nincs mit mutatni.
      .filter((entry) => entry.count > 0);

  const material = (side) =>
    PIECE_ORDER.reduce((sum, type) => sum + counts[side][type] * PIECE_VALUE[type], 0);

  return {
    capturedByWhite: missing("b"),
    capturedByBlack: missing("w"),
    advantage: material("w") - material("b"),
  };
}
