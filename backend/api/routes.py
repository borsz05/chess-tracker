from fastapi import APIRouter, HTTPException, Request, WebSocket
from pydantic import BaseModel

from backend.core.state import BackendState
from backend.services.ws_service import WebSocketHub

router = APIRouter(prefix="/api", tags=["api"])


class MoveRequest(BaseModel):
    uci: str


def get_backend_state(request: Request) -> BackendState:
    backend_state = getattr(request.app.state, "backend_state", None)
    if backend_state is None:
        raise HTTPException(status_code=503, detail="Backend state not initialized yet")
    return backend_state


@router.get("/health")
def health():
    return {"ok": True}


@router.get("/state")
def get_state(request: Request):
    state = get_backend_state(request)
    return state.get_state()


@router.post("/new-game")
def new_game(request: Request, fen: str | None = None):
    state = get_backend_state(request)
    return state.new_game(fen)


@router.post("/move")
def post_move(request: Request, req: MoveRequest):
    state = get_backend_state(request)

    try:
        return state.apply_uci(req.uci)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/robot/busy")
def get_robot_busy(request: Request):
    """A vision folyamat fél másodpercenként hívja. A `game_epoch` azért utazik
    együtt a `busy`-val, hogy ne kelljen külön lekérdezés: ha a szám változik,
    a vision tudja, hogy új parti indult (pl. a weboldal "Új parti" gombjáról),
    és nullázza a saját trackerét."""
    state = get_backend_state(request)
    with state._lock:
        busy = state.robot_busy
        game_epoch = state.game_epoch
    return {"busy": busy, "game_epoch": game_epoch}


@router.get("/robot/best-move")
def get_robot_best_move(request: Request):
    state = get_backend_state(request)
    return state.get_robot_best_move_payload()


@router.websocket("/ws/state")
async def websocket_state(websocket: WebSocket):
    backend_state: BackendState | None = getattr(websocket.app.state, "backend_state", None)
    ws_hub: WebSocketHub = websocket.app.state.ws_hub

    if backend_state is None:
        await websocket.close(code=1013, reason="Backend state not initialized yet")
        return

    await ws_hub.connect(websocket)

    try:
        await websocket.send_json({
            "type": "state",
            "payload": backend_state.get_state(),
        })

        while True:
            msg = await websocket.receive_text()
            if msg == "ping":
                await websocket.send_json({"type": "pong"})
    except Exception:
        # A szokásos eset a WebSocketDisconnect (az is Exception), de bármi
        # okból szakad meg a kapcsolat, a hubból ki kell venni.
        await ws_hub.disconnect(websocket)
