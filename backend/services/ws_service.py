from __future__ import annotations

import asyncio
from typing import Set

from fastapi import WebSocket


class WebSocketHub:
    def __init__(self):
        self._connections: Set[WebSocket] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = asyncio.Lock()

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._connections.discard(websocket)

    async def broadcast_state(self, state: dict) -> None:
        async with self._lock:
            connections = list(self._connections)

        if not connections:
            return

        dead = []
        payload = {"type": "state", "payload": state}

        for ws in connections:
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)

        if dead:
            async with self._lock:
                for ws in dead:
                    self._connections.discard(ws)

    def notify_state_changed(self, state: dict) -> None:
        if self._loop is None:
            return

        asyncio.run_coroutine_threadsafe(
            self.broadcast_state(state),
            self._loop,
        )