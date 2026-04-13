import {
  formatLineWithMoveNumbers,
  formatTopLineScore,
  formatEvalBarScore,
  normalizeScore,
} from "../utils/format.js";
import { emptyElement } from "../utils/dom.js";
import { isGameFinished } from "../utils/game.js";

function splitLineForOverflow(span, fullText) {
  const tokens = (fullText || "").split(/\s+/).filter(Boolean);
  if (tokens.length === 0) {
    return "";
  }

  let lastFit = "";
  for (let i = 0; i < tokens.length; i += 1) {
    const candidate = lastFit ? `${lastFit} ${tokens[i]}` : tokens[i];
    const preview = i < tokens.length - 1 ? `${candidate} ...` : candidate;

    span.textContent = preview;

    const fits = span.scrollWidth <= span.clientWidth + 0.5;
    if (fits) {
      lastFit = candidate;
    } else {
      return lastFit ? `${lastFit} ...` : `${tokens[0]} ...`;
    }
  }

  return lastFit;
}

export function reorderTopLinesForSide(topLines, sideToMove) {
  if (!Array.isArray(topLines)) return [];

  const arr = [...topLines];
  const scoreOrZero = (line) => normalizeScore(line).numeric;

  if (sideToMove === "b") {
    arr.sort((a, b) => scoreOrZero(a) - scoreOrZero(b));
  } else {
    arr.sort((a, b) => scoreOrZero(b) - scoreOrZero(a));
  }

  return arr;
}

export function buildTopLinesViewModel(state) {
  const sideToMove = state?.side_to_move ?? "w";
  const fullmoveNumber = state?.fullmove_number ?? 1;
  const rawLines = Array.isArray(state?.top_lines) ? state.top_lines : [];
  const orderedLines = reorderTopLinesForSide(rawLines, sideToMove);

  if (orderedLines.length > 0) {
    const main = orderedLines[0];
    const { numeric: evalValue, label: evalLabel } = formatEvalBarScore(main);

    let bestMove = null;
    if (typeof main.line_san === "string" && main.line_san.length > 0) {
      bestMove = main.line_san.split(" ")[0];
    }

    return {
      lines: orderedLines,
      sideToMove,
      fullmoveNumber,
      evalEntry: main,
      evalLabel,
      evalValue,
      bestMove,
      analysisPending: !!state?.analysis_pending,
      finished: isGameFinished(state),
      state,
    };
  }

  return {
    lines: [],
    sideToMove,
    fullmoveNumber,
    evalEntry: 0,
    evalLabel: "0.0",
    evalValue: 0,
    bestMove: null,
    analysisPending: !!state?.analysis_pending,
    finished: isGameFinished(state),
    state,
  };
}

export function updateTopLinesUI(viewModel) {
  const container = document.getElementById("top-lines");
  if (!container) return;

  emptyElement(container);

  const lines = viewModel?.lines || [];
  const sideToMove = viewModel?.sideToMove ?? "w";
  const fullmoveNumber = viewModel?.fullmoveNumber ?? 1;

  if (!lines.length) {
    return;
  }

  container.classList.remove("top-lines-game-over");

  lines.forEach((line) => {
    const { numeric: v, label: txt } = formatTopLineScore(line);

    const sanWithNumbers = formatLineWithMoveNumbers(
      line.line_san || "",
      sideToMove,
      fullmoveNumber
    );

    const row = document.createElement("div");
    row.className = "top-line";

    const scoreSpan = document.createElement("span");
    scoreSpan.className = "top-score";
    scoreSpan.textContent = txt;

    const whiteLeading = v >= 0;
    scoreSpan.classList.toggle("score-white-lead", whiteLeading);
    scoreSpan.classList.toggle("score-black-lead", !whiteLeading);

    const moveSpan = document.createElement("span");
    moveSpan.className = "top-move-line";
    moveSpan.textContent = sanWithNumbers || "";

    row.appendChild(scoreSpan);
    row.appendChild(moveSpan);
    container.appendChild(row);

    const truncated = splitLineForOverflow(moveSpan, sanWithNumbers || "");
    moveSpan.textContent = truncated;
  });
}
