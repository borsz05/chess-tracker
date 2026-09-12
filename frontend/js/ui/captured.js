import { getEl, emptyElement } from "../utils/dom.js";

const SVG_NS = "http://www.w3.org/2000/svg";
import { materialFromFen } from "../utils/material.js";
import { PIECE_PATHS } from "./piece-paths.js";

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

/**
 * Egy bábu sziluettje: a Cburnett-alakzat, de lapos kitöltéssel és a belső
 * vonalakkal KIVÁGÁSKÉNT (lásd tools/make_piece_paths.py). A színeket a
 * style.css adja, itt csak a geometria és a szerepek vannak.
 */
function buildIcon(type, side) {
  const paths = PIECE_PATHS[type];
  if (!paths) return null;

  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", "0 0 45 45");
  svg.setAttribute("class", `cap-icon cap-icon-${side}`);
  svg.setAttribute("aria-hidden", "true");

  for (const entry of paths) {
    const el = document.createElementNS(SVG_NS, "path");
    el.setAttribute("d", entry.d);
    el.setAttribute("data-role", entry.role);
    if (entry["stroke-linecap"]) el.setAttribute("stroke-linecap", entry["stroke-linecap"]);
    if (entry["stroke-linejoin"]) el.setAttribute("stroke-linejoin", entry["stroke-linejoin"]);
    svg.appendChild(el);
  }

  return svg;
}

/** Egy bábutípus csoportja: az azonos bábuk egymásra csúsztatva. */
function buildGroup(type, count, side) {
  const group = document.createElement("span");
  group.className = "cap-group";

  for (let i = 0; i < count; i++) {
    const icon = buildIcon(type, side);
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
