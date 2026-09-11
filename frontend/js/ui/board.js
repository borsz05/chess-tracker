import { getCurrentState } from "../app/state.js";
import { isGameFinished } from "../utils/game.js";

let board = null;

const SVG_NS = "http://www.w3.org/2000/svg";

export function getBoard() {
  return board;
}

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

function onUserDrop(source, target, onDropHandler) {
  if (source === target) return "snapback";
  if (isGameFinished(getCurrentState())) return "snapback";

  if (typeof onDropHandler === "function") {
    onDropHandler(source, target);
  }
}

export function initBoard(onDropHandler) {
  const boardElement = document.getElementById("board");

  board = Chessboard(boardElement, {
    position: "start",
    draggable: true,
    moveSpeed: "fast",
    snapbackSpeed: 150,
    snapSpeed: 100,
    pieceTheme: (piece) =>
      `assets/chessboardjs-1.0.0/img/chesspieces/chesscom_real/${piece.toLowerCase()}.png`,
    onDrop: (source, target) => onUserDrop(source, target, onDropHandler),
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

let lastHighlightedUci = null;

export function highlightLastMove(uci) {
  lastHighlightedUci = typeof uci === "string" && uci.length >= 4 ? uci : null;

  const $board = window.jQuery ? window.jQuery("#board") : null;
  if (!$board) return;

  $board.find(`.${SQUARE_CLASS}`).removeClass(HIGHLIGHT_CLASS);
  if (!lastHighlightedUci) return;

  $board.find(`.square-${lastHighlightedUci.slice(0, 2)}`).addClass(HIGHLIGHT_CLASS);
  $board.find(`.square-${lastHighlightedUci.slice(2, 4)}`).addClass(HIGHLIGHT_CLASS);
}

/* ── A motor javaslata: halvány nyíl ─────────────────────────
   Az SVG viewBoxa 0..100 mindkét irányban (preserveAspectRatio="none"),
   így a mezőközepek egyszerű százalékok — a tábla mérete változhat, a nyíl
   együtt mozog vele. */

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

/* A nyíl méretezése a lichess chessgroundból (src/svg.ts). Ott a viewBox
   egysége EGY mező, itt a tábla 0..100, tehát egy mező 12.5 — minden
   chessground-érték ezzel szorzódik.

   chessground:  stroke-width = lineWidth / 64        (lineWidth alapból 10)
                 arrowMargin  = (shorten ? 20 : 10) / 64
   A margó a CÉLNÁL rövidíti a vonalat; a `shorten` akkor igaz, ha a
   célmezőn áll bábu — így a nyílhegy megáll a bábu előtt, nem takarja el. */
const SQUARE_UNITS = 12.5;
const ARROW_MARGIN_OCCUPIED = (20 / 64) * SQUARE_UNITS;
const ARROW_MARGIN_EMPTY = (10 / 64) * SQUARE_UNITS;

/* A nyílhegy chessground-marker: a path és a marker méretei szó szerint
   onnan valók (markerWidth/Height 4, refX 2.05, refY 2, "M0,0 V4 L3,2 Z").
   markerUnits alapból "strokeWidth", ezért a hegy a szárral együtt nő. */
function buildArrowHeadDefs() {
  const defs = document.createElementNS(SVG_NS, "defs");
  const marker = document.createElementNS(SVG_NS, "marker");

  marker.setAttribute("id", "suggestion-arrowhead");
  marker.setAttribute("orient", "auto");
  marker.setAttribute("markerWidth", "4");
  marker.setAttribute("markerHeight", "4");
  marker.setAttribute("refX", "2.05");
  marker.setAttribute("refY", "2");

  const path = document.createElementNS(SVG_NS, "path");
  path.setAttribute("class", "suggestion-head");
  path.setAttribute("d", "M0,0 V4 L3,2 Z");

  marker.appendChild(path);
  defs.appendChild(marker);
  return defs;
}

function isSquareOccupied(square) {
  if (!board) return false;
  const position = board.position();
  return !!(position && position[square]);
}

export function drawSuggestionArrow(uci) {
  const svg = document.getElementById("board-arrow");
  if (!svg) return;

  lastSuggestionUci = typeof uci === "string" && uci.length >= 4 ? uci : null;

  while (svg.firstChild) svg.removeChild(svg.firstChild);
  if (!lastSuggestionUci || !board) return;

  const orientation = board.orientation();
  const fromSquare = lastSuggestionUci.slice(0, 2);
  const toSquare = lastSuggestionUci.slice(2, 4);
  const from = squareCenter(fromSquare, orientation);
  const to = squareCenter(toSquare, orientation);
  if (!from || !to) return;

  const dx = to.x - from.x;
  const dy = to.y - from.y;
  if (Math.hypot(dx, dy) < 0.001) return;

  const angle = Math.atan2(dy, dx);
  const margin = isSquareOccupied(toSquare)
    ? ARROW_MARGIN_OCCUPIED
    : ARROW_MARGIN_EMPTY;

  const shaft = document.createElementNS(SVG_NS, "line");
  shaft.setAttribute("class", "suggestion");
  shaft.setAttribute("x1", from.x.toFixed(2));
  shaft.setAttribute("y1", from.y.toFixed(2));
  shaft.setAttribute("x2", (to.x - Math.cos(angle) * margin).toFixed(2));
  shaft.setAttribute("y2", (to.y - Math.sin(angle) * margin).toFixed(2));
  shaft.setAttribute("marker-end", "url(#suggestion-arrowhead)");

  svg.appendChild(buildArrowHeadDefs());
  svg.appendChild(shaft);
}

/**
 * A jobb oszlop magasságát a tábla VALÓDI magasságához igazítja (a
 * chessboard.js a mezőméretet egészre kerekíti, ezért a tábla pár pixellel
 * alacsonyabb a CSS-ben megadott szélességnél). Csak egy CSS-változót ír,
 * a tábla méretét nem érinti — nem tud visszacsatolni a ResizeObserverbe.
 */
function syncColumnHeight() {
  const inner = document.querySelector("#board .board-b72b1");
  const height = inner?.getBoundingClientRect().height;
  if (!height) return;
  document.documentElement.style.setProperty("--board-height", `${Math.round(height)}px`);
}

/** Újrarajzolja a kiemelést és a nyilat — pl. a tábla átrajzolása után. */
export function refreshBoardDecorations() {
  highlightLastMove(lastHighlightedUci);
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

  highlightLastMove(nextState?.last_move?.uci ?? null);
  // A chessboard.js animáció közben újraépíti a mezőket; a kiemelést az
  // animáció után is ki kell tenni.
  setTimeout(() => highlightLastMove(nextState?.last_move?.uci ?? null), 260);
}
