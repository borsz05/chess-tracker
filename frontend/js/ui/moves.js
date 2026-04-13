import { getEl, emptyElement } from "../utils/dom.js";

function createMoveRow(move) {
  const sanText = move.san || move.uci;

  const li = document.createElement("li");
  li.className = "move-row";
  li.dataset.plyIndex = String(move.ply_index);

  const moveNo = document.createElement("span");
  moveNo.className = "move-no";

  const white = document.createElement("span");
  white.className = "move-white";

  const black = document.createElement("span");
  black.className = "move-black";

  if (move.color === "w") {
    moveNo.textContent = `${move.move_no}.`;
    white.textContent = sanText;
    black.textContent = "";
  } else {
    moveNo.textContent = `${move.move_no}...`;
    white.textContent = "";
    black.textContent = sanText;
  }

  li.appendChild(moveNo);
  li.appendChild(white);
  li.appendChild(black);

  return li;
}

function fullRender(list, moves) {
  emptyElement(list);

  let lastLi = null;

  for (const move of moves) {
    if (move.color === "w") {
      const li = createMoveRow(move);
      list.appendChild(li);
      lastLi = li;
    } else {
      if (!lastLi) {
        const li = createMoveRow(move);
        list.appendChild(li);
        lastLi = li;
      } else {
        const blackSpan = lastLi.querySelector(".move-black");
        if (blackSpan) {
          blackSpan.textContent = move.san || move.uci;
        }
      }
    }
  }

  highlightLastMove(list, lastLi);
}

function highlightLastMove(list, lastLi) {
  list.querySelectorAll("li").forEach((li) => li.classList.remove("last-move"));

  if (lastLi) {
    lastLi.classList.add("last-move");
    lastLi.scrollIntoView({ block: "nearest" });
  }
}

function canAppendOnly(prevMoves, nextMoves) {
  if (!Array.isArray(prevMoves) || !Array.isArray(nextMoves)) return false;
  if (nextMoves.length !== prevMoves.length + 1) return false;

  for (let i = 0; i < prevMoves.length; i++) {
    if (
      prevMoves[i]?.ply_index !== nextMoves[i]?.ply_index ||
      prevMoves[i]?.uci !== nextMoves[i]?.uci ||
      prevMoves[i]?.san !== nextMoves[i]?.san
    ) {
      return false;
    }
  }

  return true;
}

export function updateMovesUI(moves, prevMoves = []) {
  const list = getEl("moves-list");
  if (!list) return;

  if (!moves?.length) {
    emptyElement(list);
    return;
  }

  if (!canAppendOnly(prevMoves, moves)) {
    fullRender(list, moves);
    return;
  }

  const lastMove = moves[moves.length - 1];
  let lastLi = list.lastElementChild;

  if (lastMove.color === "w") {
    const li = createMoveRow(lastMove);
    list.appendChild(li);
    lastLi = li;
  } else {
    if (!lastLi) {
      const li = createMoveRow(lastMove);
      list.appendChild(li);
      lastLi = li;
    } else {
      const blackSpan = lastLi.querySelector(".move-black");
      if (blackSpan) {
        blackSpan.textContent = lastMove.san || lastMove.uci;
      }
    }
  }

  highlightLastMove(list, lastLi);
}