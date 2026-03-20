import { formatEvalBarScore } from "../utils/format.js";
import { getEl, setText } from "../utils/dom.js";

export function updateEvalUI(viewModel) {
  const barFill = getEl("eval-bar-fill");
  const barScore = getEl("eval-bar-score");

  const evalEntry = viewModel?.evalEntry ?? 0;
  const bestMove = viewModel?.bestMove ?? null;
  const forcedLabel = viewModel?.evalLabel ?? null;

  const { numeric: value, label: fallbackLabel } = formatEvalBarScore(evalEntry);
  const label = forcedLabel ?? fallbackLabel;

  if (barFill) {
    const clamped = Math.max(-5, Math.min(5, value));
    const whitePercent = 50 + (clamped / 10) * 100;
    barFill.style.height = `${whitePercent}%`;
  }

  setText(barScore, label);

  if (barScore) {
    barScore.dataset.bestMove = bestMove || "";
    const whiteAdv = value >= 0;
    barScore.classList.toggle("white-adv", whiteAdv);
    barScore.classList.toggle("black-adv", !whiteAdv);
  }
}