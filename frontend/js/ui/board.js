import { kingSquareFromFen } from "../utils/game.js";
import { pieceImageUrl } from "./piece-theme.js";

let board = null;

const SVG_NS = "http://www.w3.org/2000/svg";

export function uciToBoardMove(uci) {
  if (typeof uci !== "string" || uci.length < 4) return null;

  const from = uci.slice(0, 2);
  const to = uci.slice(2, 4);
  const promotion = uci.length > 4 ? uci.slice(4, 5) : "";
  const promoSuffix = promotion ? promotion.toUpperCase() : "";

  return `${from}-${to}${promoSuffix}`;
}

function animateFromNewState(previousState, newState) {
  if (!board) return false;

  const prevMoves = previousState?.moves?.length || 0;
  const newMoves = newState?.moves?.length || 0;

  if (newMoves === 0) {
    board.position(newState.fen, true);
    return true;
  }

  if (newMoves !== prevMoves + 1) return false;

  const newMove = newState.moves[newMoves - 1];
  const moveStr = uciToBoardMove(newMove?.uci);
  if (!moveStr) return false;

  try {
    board.move(moveStr);
    setTimeout(() => board.position(newState.fen, false), 150);
    return true;
  } catch (err) {
    console.warn("Nem sikerült animálni a lépést:", err);
    return false;
  }
}

export function initBoard() {
  const boardElement = document.getElementById("board");

  board = Chessboard(boardElement, {
    position: "start",
    moveSpeed: "fast",
    pieceTheme: pieceImageUrl,
  });

  // A chessboard.js egyszer, induláskor méri meg a konténert, és soha többé.
  // Ha az oldal olyankor töltődik be, amikor az ablaknak még nincs magassága
  // (háttérfül, még ki nem nyitott panel), a 82vh = 0 -> a tábla 0 px-es
  // mezőkkel épül fel, és üresen marad. A ResizeObserver ezt orvosolja: a
  // konténer minden méretváltozásánál újraépítjük a táblát, és visszatesszük
  // a kiemelést meg a nyilat.
  observeBoardSize(boardElement);
  syncColumnHeight();
}

let boardResizeObserver = null;
let lastObservedWidth = 0;

function observeBoardSize(boardElement) {
  const rerender = () => {
    if (!board) return;
    board.resize();
    refreshBoardDecorations();
    syncColumnHeight();
  };

  if (typeof ResizeObserver === "function") {
    boardResizeObserver?.disconnect();
    boardResizeObserver = new ResizeObserver((entries) => {
      const width = Math.round(entries[0]?.contentRect?.width ?? 0);
      if (width === lastObservedWidth) return;
      lastObservedWidth = width;
      rerender();
    });
    boardResizeObserver.observe(boardElement);
    return;
  }

  window.addEventListener("resize", rerender);
}

/* ── Az utolsó lépés kiemelése ───────────────────────────────
   A chessboard.js 1.0.0-nak nincs erre API-ja; a könyvtár SAJÁT hivatalos
   példái (chessboardjs.com/examples #5004, #5005) is pontosan ezt csinálják:
   osztályt tesznek a `.square-<mező>` elemre, és CSS `box-shadow: inset`
   rajzolja a keretet. A `squareClass` a példákból vett belső osztálynév,
   ezzel lehet egy mozdulattal leszedni az összes korábbi kiemelést. */

const SQUARE_CLASS = "square-55d63";
const HIGHLIGHT_CLASS = "highlight-last-move";
const CHECK_CLASS = "in-check";

let lastHighlightedUci = null;
let lastCheckSquare = null;

export function highlightLastMove(uci) {
  lastHighlightedUci = typeof uci === "string" && uci.length >= 4 ? uci : null;

  const $board = window.jQuery ? window.jQuery("#board") : null;
  if (!$board) return;

  $board.find(`.${SQUARE_CLASS}`).removeClass(HIGHLIGHT_CLASS);
  if (!lastHighlightedUci) return;

  $board.find(`.square-${lastHighlightedUci.slice(0, 2)}`).addClass(HIGHLIGHT_CLASS);
  $board.find(`.square-${lastHighlightedUci.slice(2, 4)}`).addClass(HIGHLIGHT_CLASS);
}

/* ── Sakk: piros derengés a király mezőjén ───────────────────
   A mezőt a FEN-ből számoljuk, mert a backend csak azt küldi, hogy VAN
   sakk, azt nem, hogy hol — sakkban viszont mindig a lépni következő fél
   királya áll sakkban, tehát a FEN-ből egyértelmű. */

function applyCheckClass() {
  const $board = window.jQuery ? window.jQuery("#board") : null;
  if (!$board) return;

  $board.find(`.${SQUARE_CLASS}`).removeClass(CHECK_CLASS);
  if (!lastCheckSquare) return;

  $board.find(`.square-${lastCheckSquare}`).addClass(CHECK_CLASS);
}

export function highlightCheck(fen, isCheck) {
  lastCheckSquare = isCheck ? kingSquareFromFen(fen) : null;
  applyCheckClass();
}

/* ── A motor javaslata: halvány nyíl ─────────────────────────
   Az SVG viewBoxa 0..100 mindkét irányban (preserveAspectRatio="none"),
   így a mezőközepek egyszerű százalékok.

   Az SVG-t viszont NEM lehet a #board-wrap egészére kifeszíteni, pedig
   kézenfekvő lenne. A chessboard.js EGÉSZRE KEREKÍTI a mezőméretet, a
   tábla tehát majdnem mindig keskenyebb a konténerénél, ráadásul van egy
   1px-es kerete is (box-sizing: content-box, tehát a keret KÍVÜLRE nő).
   Mérve: konténer 590.39px, mezőméret 73px -> a valódi rács 8*73 = 584px.
   A teljes konténerre feszített viewBox így 73.80px-es „mezőkkel" számol
   73 helyett, és a hiba vonalanként halmozódik: az a-vonalon még -0.6px,
   a h-vonalon már +5.0px — a nyíl jobbra csúszik a mező közepétől.

   Ezért az SVG-t a tábla VALÓDI rácsára illesztjük, a kereten belülre. */

function syncArrowViewport(svg) {
  const wrap = document.getElementById("board-wrap");
  const a1 = document.querySelector("#board .square-a1");
  const h8 = document.querySelector("#board .square-h8");
  if (!wrap || !a1 || !h8) return;

  // A rácsot MAGUKBÓL A MEZŐKBŐL olvassuk ki, nem a tábla elemének a
  // méretéből: így nem kell a kerettel, a box-sizinggal vagy az oldal
  // nagyításával számolni — egyik sem tudja elrontani. Az a1 és a h8
  // átellenes sarok mindkét tájolásban, ezért elég a kettő min/maxa.
  const wrapRect = wrap.getBoundingClientRect();
  const one = a1.getBoundingClientRect();
  const other = h8.getBoundingClientRect();
  if (!one.width) return;

  const left = Math.min(one.left, other.left);
  const top = Math.min(one.top, other.top);
  const width = Math.max(one.right, other.right) - left;
  const height = Math.max(one.bottom, other.bottom) - top;

  svg.style.left = `${left - wrapRect.left}px`;
  svg.style.top = `${top - wrapRect.top}px`;
  svg.style.width = `${width}px`;
  svg.style.height = `${height}px`;
}

function squareCenter(square, orientation) {
  if (typeof square !== "string" || square.length < 2) return null;

  const file = square.charCodeAt(0) - 97; // a..h -> 0..7
  const rank = Number(square[1]) - 1; // 1..8 -> 0..7
  if (file < 0 || file > 7 || rank < 0 || rank > 7) return null;

  const flipped = orientation === "black";
  const col = flipped ? 7 - file : file;
  const row = flipped ? rank : 7 - rank;

  return { x: (col + 0.5) * 12.5, y: (row + 0.5) * 12.5 };
}

let lastSuggestionUci = null;

/* A nyíl geometriája a CHESS.COM-éból, a böngésző fejlesztői eszközeivel
   kinyert polygonból visszafejtve. Ők EGYETLEN hétpontos polygonnal rajzolnak
   (nem vonal + marker párossal), ráadásul ugyanabban a 0..100-as viewBoxban,
   mint mi — így az értékek közvetlenül átvehetők.

   A mért arányok (egy mező = 12.5 egység):
     szár szélessége  2.75  = 0.220 mező
     hegy szélessége  6.50  = 0.520 mező
     hegy hossza      4.50  = 0.360 mező
     indulási hézag   4.50  = 0.360 mező, a FORRÁS közepétől

   Két dologban tér el a korábbi, chessground-alapú változatunktól: a szár
   vastagabb, a hegy pedig kisebb; és a hegy CSÚCSA pontosan a célmező
   közepén áll, vagyis a rövidítés a forrás oldalán van, nem a célnál.

   A lenti pontsorrend a d2d4 példa mind a hét pontját pontosan visszaadja. */
const ARROW_SHAFT_HALF = 1.375;   /* 2.75 / 2 */
const ARROW_HEAD_HALF  = 3.25;    /* 6.50 / 2 */
const ARROW_HEAD_LEN   = 4.5;
const ARROW_START_GAP  = 4.5;

function buildArrowPoints(from, to) {
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const len = Math.hypot(dx, dy);
  if (len < 0.001) return null;

  const ux = dx / len;
  const uy = dy / len;
  const px = -uy;   // a haladási irányra merőleges egységvektor
  const py = ux;

  const sx = from.x + ux * ARROW_START_GAP;   // a szár kezdete
  const sy = from.y + uy * ARROW_START_GAP;
  const bx = to.x - ux * ARROW_HEAD_LEN;      // a hegy alapja
  const by = to.y - uy * ARROW_HEAD_LEN;

  const points = [
    [sx + px * ARROW_SHAFT_HALF, sy + py * ARROW_SHAFT_HALF],
    [bx + px * ARROW_SHAFT_HALF, by + py * ARROW_SHAFT_HALF],
    [bx + px * ARROW_HEAD_HALF,  by + py * ARROW_HEAD_HALF],
    [to.x, to.y],                              // a csúcs: a célmező közepe
    [bx - px * ARROW_HEAD_HALF,  by - py * ARROW_HEAD_HALF],
    [bx - px * ARROW_SHAFT_HALF, by - py * ARROW_SHAFT_HALF],
    [sx - px * ARROW_SHAFT_HALF, sy - py * ARROW_SHAFT_HALF],
  ];

  return points.map(([x, y]) => `${x.toFixed(3)},${y.toFixed(3)}`).join(" ");
}

/* LÓLÉPÉS: a chess.com nem átlós nyilat húz, hanem MEGTÖRTET — előbb a hosszú
   szár (a két mezős irány), aztán derékszögben a rövid (az egy mezős). A szár
   vastagsága, a hegy mérete és az indulási hézag ugyanaz, mint az egyenesnél;
   csak egy töréspont kerül bele, éles (nem lekerekített) sarokkal.

   A sarokpontok a két eltolt oldalegyenes metszéspontjai. Mivel a törés
   pontosan 90 fokos, u1 merőleges u2-re, így a metszéspont egyetlen
   vetítéssel megkapható — nem kell általános egyenes-metszést számolni.

   Ellenőrizve: a g1f3 példára ez a kód mind a KILENC pontot 0.00e+00
   eltéréssel adja vissza, a chess.com transform="rotate(180 81.25 93.75)"-
   ának feloldása után. */

function knightBend(fromSquare, toSquare, orientation) {
  const ff = fromSquare.charCodeAt(0) - 97;
  const fr = Number(fromSquare[1]);
  const tf = toSquare.charCodeAt(0) - 97;
  const tr = Number(toSquare[1]);

  const df = Math.abs(tf - ff);
  const dr = Math.abs(tr - fr);
  if (!((df === 1 && dr === 2) || (df === 2 && dr === 1))) return null;

  // A töréspont a KÉT mezős tengely mentén van: arra megyünk előbb.
  const bendFile = dr === 2 ? ff : tf;
  const bendRank = dr === 2 ? tr : fr;
  return squareCenter(`${"abcdefgh"[bendFile]}${bendRank}`, orientation);
}

function buildKnightArrowPoints(from, bend, to) {
  const u1x = Math.sign(bend.x - from.x) || 0;
  const u1y = Math.sign(bend.y - from.y) || 0;
  const u2x = Math.sign(to.x - bend.x) || 0;
  const u2y = Math.sign(to.y - bend.y) || 0;
  const p1x = -u1y, p1y = u1x;
  const p2x = -u2y, p2y = u2x;

  const sx = from.x + u1x * ARROW_START_GAP;
  const sy = from.y + u1y * ARROW_START_GAP;
  const bx = to.x - u2x * ARROW_HEAD_LEN;
  const by = to.y - u2y * ARROW_HEAD_LEN;

  // A két eltolt oldalegyenes metszéspontja (u1 merőleges u2-re).
  const cornerOn = (side) => {
    const q1x = sx + p1x * ARROW_SHAFT_HALF * side;
    const q1y = sy + p1y * ARROW_SHAFT_HALF * side;
    const q2x = bx + p2x * ARROW_SHAFT_HALF * side;
    const q2y = by + p2y * ARROW_SHAFT_HALF * side;
    const a = (q2x - q1x) * u1x + (q2y - q1y) * u1y;
    return [q1x + a * u1x, q1y + a * u1y];
  };

  const points = [
    [sx + p1x * ARROW_SHAFT_HALF, sy + p1y * ARROW_SHAFT_HALF],
    cornerOn(+1),
    [bx + p2x * ARROW_SHAFT_HALF, by + p2y * ARROW_SHAFT_HALF],
    [bx + p2x * ARROW_HEAD_HALF,  by + p2y * ARROW_HEAD_HALF],
    [to.x, to.y],
    [bx - p2x * ARROW_HEAD_HALF,  by - p2y * ARROW_HEAD_HALF],
    [bx - p2x * ARROW_SHAFT_HALF, by - p2y * ARROW_SHAFT_HALF],
    cornerOn(-1),
    [sx - p1x * ARROW_SHAFT_HALF, sy - p1y * ARROW_SHAFT_HALF],
  ];

  return points.map(([x, y]) => `${x.toFixed(3)},${y.toFixed(3)}`).join(" ");
}

export function drawSuggestionArrow(uci) {
  const svg = document.getElementById("board-arrow");
  if (!svg) return;

  lastSuggestionUci = typeof uci === "string" && uci.length >= 4 ? uci : null;

  // A tábla átméreteződhetett az előző rajzolás óta.
  syncArrowViewport(svg);

  while (svg.firstChild) svg.removeChild(svg.firstChild);
  if (!lastSuggestionUci || !board) return;

  const orientation = board.orientation();
  const fromSquare = lastSuggestionUci.slice(0, 2);
  const toSquare = lastSuggestionUci.slice(2, 4);
  const from = squareCenter(fromSquare, orientation);
  const to = squareCenter(toSquare, orientation);
  if (!from || !to) return;

  const bend = knightBend(fromSquare, toSquare, orientation);
  const points = bend
    ? buildKnightArrowPoints(from, bend, to)
    : buildArrowPoints(from, to);
  if (!points) return;

  const arrow = document.createElementNS(SVG_NS, "polygon");
  arrow.setAttribute("class", "suggestion");
  arrow.setAttribute("points", points);

  svg.appendChild(arrow);
}

/**
 * A jobb oszlop magasságát a tábla VALÓDI magasságához igazítja (a
 * chessboard.js a mezőméretet egészre kerekíti, ezért a tábla pár pixellel
 * alacsonyabb a CSS-ben megadott szélességnél). Csak egy CSS-változót ír,
 * a tábla méretét nem érinti — nem tud visszacsatolni a ResizeObserverbe.
 */
function syncColumnHeight() {
  // A jobb oszlop a TELJES bal oldali blokkhoz igazodik: tábla + a fölötte
  // és alatta futó leütött-bábu sávok. Ha a sávok valamiért hiányoznának,
  // visszaesünk a puszta tábla magasságára.
  const stack = document.querySelector(".board-stack");
  const inner = document.querySelector("#board .board-b72b1");
  const height = (stack ?? inner)?.getBoundingClientRect().height;
  if (!height) return;
  document.documentElement.style.setProperty("--board-height", `${Math.round(height)}px`);
}

/** Újrarajzolja a kiemelést és a nyilat — pl. a tábla átrajzolása után. */
export function refreshBoardDecorations() {
  highlightLastMove(lastHighlightedUci);
  applyCheckClass();
  drawSuggestionArrow(lastSuggestionUci);
}

export function syncBoardFromState(previousState, nextState, opts = {}) {
  const animateFromLast = opts.animateFromLast ?? true;
  if (!board || !nextState?.fen) return;

  const animated = animateFromLast
    ? animateFromNewState(previousState, nextState)
    : false;

  if (!animated) {
    board.position(nextState.fen, true);
  }

  const checkNow = !!nextState?.status?.is_check;

  highlightLastMove(nextState?.last_move?.uci ?? null);
  highlightCheck(nextState?.fen ?? null, checkNow);
  // A chessboard.js animáció közben újraépíti a mezőket; a kiemelést az
  // animáció után is ki kell tenni.
  setTimeout(() => {
    highlightLastMove(nextState?.last_move?.uci ?? null);
    highlightCheck(nextState?.fen ?? null, checkNow);
  }, 260);
}
