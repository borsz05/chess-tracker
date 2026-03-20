import { fetchState, postMove, postNewGame } from "../api/api.js";
import { StateSocket } from "../api/socket.js";
import { getCurrentState, getConnectionState, setConnectionState, setCurrentState } from "./state.js";
import {
  selectBoardSignature,
  selectMoves,
  selectMovesSignature,
  selectStatusSignature,
  selectTopLinesSignature,
} from "./selectors.js";
import { initBoard, syncBoardFromState } from "../ui/board.js";
import { updateEvalUI } from "../ui/eval.js";
import { updateMovesUI } from "../ui/moves.js";
import { updateStatusUI } from "../ui/status.js";
import { buildTopLinesViewModel, updateTopLinesUI } from "../ui/toplines.js";
import { showToast } from "../ui/toast.js";
import { getEl, setText } from "../utils/dom.js";
import { isGameFinished } from "../utils/game.js";

let stateSocket = null;

function renderConnectionOnly() {
  updateStatusUI(getCurrentState(), getConnectionState());
}

function renderSelective(previousState, nextState, opts = {}) {
  const prevBoardSig = selectBoardSignature(previousState);
  const nextBoardSig = selectBoardSignature(nextState);

  if (prevBoardSig !== nextBoardSig) {
    syncBoardFromState(previousState, nextState, {
      animateFromLast: opts.animateFromLast ?? true,
    });
  }

  const prevTopSig = selectTopLinesSignature(previousState);
  const nextTopSig = selectTopLinesSignature(nextState);

  if (prevTopSig !== nextTopSig) {
    const topLinesView = buildTopLinesViewModel(nextState);
    updateEvalUI(topLinesView);
    updateTopLinesUI(topLinesView);
  }

  const prevMovesSig = selectMovesSignature(previousState);
  const nextMovesSig = selectMovesSignature(nextState);

  if (prevMovesSig !== nextMovesSig) {
    updateMovesUI(selectMoves(nextState), selectMoves(previousState));
  }

  const prevStatusSig = selectStatusSignature(previousState);
  const nextStatusSig = selectStatusSignature(nextState);

  if (prevStatusSig !== nextStatusSig) {
    updateStatusUI(nextState, getConnectionState());
  }
}

function applyServerState(nextState, opts = {}) {
  const previousState = getCurrentState();
  setCurrentState(nextState);
  renderSelective(previousState, nextState, opts);
}

async function reloadStateFallback(opts = {}) {
  try {
    const state = await fetchState();
    applyServerState(state, opts);
  } catch (err) {
    console.error("Állapot fallback betöltési hiba:", err);
  }
}

async function handleUserMove(source, target) {
  const current = getCurrentState();
  if (isGameFinished(current)) return;

  const uci = `${source}${target}`;

  try {
    const state = await postMove(uci);
    applyServerState(state, { animateFromLast: true });
  } catch (err) {
    console.error("Lépés hiba:", err);
    await reloadStateFallback({ animateFromLast: false });
  }
}

async function handleNewGame() {
  try {
    const state = await postNewGame();
    applyServerState(state, { animateFromLast: false });
    setText(getEl("pgn-status"), "");
  } catch (err) {
    console.error("Új játék hiba:", err);
  }
}

async function handleCopyPgn() {
  const state = getCurrentState();

  try {
    if (!state?.pgn) {
      showToast("Még nincs PGN.", "error", 2000);
      return;
    }

    await navigator.clipboard.writeText(state.pgn);
    showToast("PGN vágólapra másolva.", "success", 2200);
  } catch (err) {
    console.error("PGN másolási hiba:", err);
    showToast("Hiba a PGN másolásakor.", "error", 2200);
  }
}

function bindButtons() {
  const newGameBtn = getEl("new-game-btn");
  const copyPgnBtn = getEl("copy-pgn-btn");

  if (newGameBtn) {
    newGameBtn.addEventListener("click", handleNewGame);
  }

  if (copyPgnBtn) {
    copyPgnBtn.addEventListener("click", handleCopyPgn);
  }
}

function initSocket() {
  stateSocket = new StateSocket({
    onState: (state) => {
      applyServerState(state, { animateFromLast: true });
    },
    onStatus: (status) => {
      setConnectionState(status);
      renderConnectionOnly();
    },
  });

  stateSocket.connect();
}

export async function bootstrapApp() {
  initBoard(handleUserMove);
  bindButtons();
  renderConnectionOnly();
  initSocket();

  await reloadStateFallback({ animateFromLast: false });
}