/**
 * Bábu-sziluettek a leütött bábuk kijelzéséhez.
 *
 * Saját rajzolású SVG-k, szándékosan: így nincs licenc-kérdés, és a méretük
 * meg a színük a CSS-ből vezérelhető (a fill `currentColor`).
 *
 * Mind egy 24x24-es viewBoxban ül, azonos talapzat-vonallal, hogy egymás
 * mellé téve egy sorba rendeződjenek.
 */

const PATHS = {
  // Gyalog: fej, nyak, harangtest, talapzat.
  p: "M12 3.4a3 3 0 0 1 1.7 5.5c1.5.9 2.5 2.4 2.8 4.3H7.5c.3-1.9 1.3-3.4 2.8-4.3A3 3 0 0 1 12 3.4z"
   + "M8.2 14.6h7.6c.1 2.3-.5 4-1.4 5.1H9.6c-.9-1.1-1.5-2.8-1.4-5.1z"
   + "M6.1 19.9h11.8v2.1H6.1z",

  // Futó: felső gömb, mitra egy bevágással, gallér, talapzat.
  b: "M12 2.6a1.5 1.5 0 0 1 .9 2.7c2.2 1.5 3.7 3.7 3.7 5.9 0 1.8-.9 3.1-2.1 4h-1.3l1.5-2.1-1.5-1.5"
   + "l-1.5 1.5 1.5 2.1H9.5c-1.2-.9-2.1-2.2-2.1-4 0-2.2 1.5-4.4 3.7-5.9A1.5 1.5 0 0 1 12 2.6z"
   + "M8.6 16.4h6.8c.6.9 1 1.9 1 2.6H7.6c0-.7.4-1.7 1-2.6z"
   + "M6.1 19.9h11.8v2.1H6.1z",

  // Huszár: lófej profilból, sörénnyel.
  n: "M9.6 2.8c.6 0 1.1.3 1.5.8l3.9 1.7c2.4 1 3.9 3.3 3.9 6v8.6h-9.3v-2.8c0-1.9.9-3.2 2.3-4.2"
   + "l2-1.4-3.2.8c-1.4.3-2.4 1-3.1 2.1l-2-2.6c-.7-.9-.7-2.1-.1-3l2.2-3.3c.3-1.4.9-2.7 1.9-2.7z"
   + "M6.1 19.9h11.8v2.1H6.1z",

  // Bástya: oromzat, derék, talapzat.
  r: "M5.9 3.6h2.9v1.9h2.1V3.6h2.2v1.9h2.1V3.6h2.9v4.5l-1.8 1.7v5.5l1.8 1.8v1.7H6.1v-1.7l1.8-1.8"
   + "V9.8L5.9 8.1z"
   + "M4.9 19.9h14.2v2.1H4.9z",

  // Vezér: öt ágú korona gömbökkel, test, talapzat.
  q: "M4.3 6.4a1.4 1.4 0 1 1 1.6 1.4l1.2 3.1 2-4.4a1.5 1.5 0 1 1 1.8-.1L12 9.9l1.1-3.5a1.5 1.5 0 1 1 1.8.1"
   + "l2 4.4 1.2-3.1a1.4 1.4 0 1 1 1.6-1.4c0 .7-.6 1.3-1.3 1.4l-1.5 6.2H7.1L5.6 7.8A1.4 1.4 0 0 1 4.3 6.4z"
   + "M7.5 16.4h9c.4.9.6 1.8.6 2.6H6.9c0-.8.2-1.7.6-2.6z"
   + "M5.1 19.9h13.8v2.1H5.1z",
};

/**
 * Egy bábu-sziluett SVG eleme.
 * @param {string} type  p | b | n | r | q
 * @param {string} side  "w" | "b" — csak osztályt ad, a színt a CSS dönti el
 */
export function createPieceIcon(type, side) {
  const path = PATHS[type];
  if (!path) return null;

  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("class", `cap-icon cap-icon-${side}`);
  svg.setAttribute("aria-hidden", "true");

  const el = document.createElementNS("http://www.w3.org/2000/svg", "path");
  el.setAttribute("d", path);
  svg.appendChild(el);

  return svg;
}
