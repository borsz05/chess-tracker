import { getCurrentState } from "../app/state.js";
import { isGameFinished } from "../utils/game.js";

let board = null;

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
      `assets/chessboardjs-1.0.0/img/chesspieces/chesscom_icy/${piece.toLowerCase()}.png`,
    onDrop: (source, target) => onUserDrop(source, target, onDropHandler),
  });
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
}