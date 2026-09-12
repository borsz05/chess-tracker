import { getEl, emptyElement } from "../utils/dom.js";
import { materialFromFen } from "../utils/material.js";
import { createPieceIcon } from "./piece-icons.js";

/**
 * A leütött bábuk két sávja a tábla fölött és alatt.
 *
 * A FELSŐ sáv a sötété: azt mutatja, amit a sötét ütött le, vagyis VILÁGOS
 * bábukat. Az alsó fordítva. (A tábla mindig világos nézetben áll, tehát a
 * felül lévő fél a sötét.)
 *
 * A "+N" mindig csak a VEZETŐ oldalán jelenik meg — döntetlen anyagnál
 * egyiken sem.
 */

/** Egy bábutípus csoportja: az azonos bábuk egymásra csúsztatva. */
function buildGroup(type, count, side) {
  const group = document.createElement("span");
  group.className = "cap-group";

  for (let i = 0; i < count; i++) {
    const icon = createPieceIcon(type, side);
    if (icon) group.appendChild(icon);
  }

  return group;
}

function renderStrip(el, entries, side, score) {
  emptyElement(el);
  if (!el) return;

  for (const { type, count } of entries) {
    el.appendChild(buildGroup(type, count, side));
  }

  if (score > 0) {
    const span = document.createElement("span");
    span.className = "cap-score";
    span.textContent = `+${score}`;
    el.appendChild(span);
  }
}

export function updateCapturedUI(fen) {
  const top = getEl("captured-black");
  const bottom = getEl("captured-white");
  if (!top && !bottom) return;

  const { capturedByWhite, capturedByBlack, advantage } = materialFromFen(fen);

  // A felül lévő fél a sötét: ő VILÁGOS bábukat üt le.
  renderStrip(top, capturedByBlack, "w", advantage < 0 ? -advantage : 0);
  renderStrip(bottom, capturedByWhite, "b", advantage > 0 ? advantage : 0);
}
