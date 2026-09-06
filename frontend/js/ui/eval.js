import { formatEvalBarScore } from "../utils/format.js";
import { getEl, setText } from "../utils/dom.js";
import { decisiveResult, resultBadgeText } from "../utils/game.js";

export function updateEvalUI(viewModel) {
  const barFill = getEl("eval-bar-fill");
  const barScore = getEl("eval-bar-score");

  const result = viewModel?.finished ? decisiveResult(viewModel.state) : null;

  // Lezárt parti: a sáv a győztes színével telik meg, és a centipawn-érték
  // helyén az eredmény áll. Döntetlennél felezve marad.
  if (result) {
    if (barFill) {
      barFill.style.height = result === "1-0" ? "100%" : result === "0-1" ? "0%" : "50%";
    }
    setText(barScore, resultBadgeText(result));
    if (barScore) {
      barScore.dataset.bestMove = "";
      barScore.classList.remove("white-adv", "black-adv");
      barScore.classList.add("result");
      // A felirat a sáv közepén ül, a kitöltéshez igazodó kontraszttal.
      barScore.classList.toggle("on-white", result === "1-0");
      barScore.classList.toggle("on-black", result === "0-1");
      barScore.classList.toggle("on-draw", result === "1/2-1/2");
    }
    return;
  }

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
    barScore.classList.remove("result", "on-white", "on-black", "on-draw");
    barScore.classList.toggle("white-adv", whiteAdv);
    barScore.classList.toggle("black-adv", !whiteAdv);
  }
}
