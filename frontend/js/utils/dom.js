export function getEl(id) {
  return document.getElementById(id);
}

export function setText(el, text) {
  if (el) el.textContent = text ?? "";
}

export function emptyElement(el) {
  if (!el) return;
  while (el.firstChild) {
    el.removeChild(el.firstChild);
  }
}