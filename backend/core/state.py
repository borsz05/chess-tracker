from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, Optional

import chess

from chess_logic import Game
from backend.services.engine_service import AnalysisSnapshot, EngineAnalysisService
from backend.services.robot_service import best_move_payload_from_top_lines


@dataclass(slots=True)
class BackendStateSettings:
    start_fen: str
    stockfish_path: str
    deep_depth: int
    deep_multipv: int


class BackendState:
    def __init__(self, settings):
        self.settings = BackendStateSettings(
          start_fen=settings.start_fen,
          stockfish_path=settings.stockfish_path,
          deep_depth=settings.deep_depth,
          deep_multipv=settings.deep_multipv,
        )

        self._lock = threading.Lock()
        self._state_notifier: Optional[Callable[[dict], None]] = None

        self.game = Game(self.settings.start_fen)

        self.engine_service = EngineAnalysisService(
            engine_path=self.settings.stockfish_path,
            default_depth=self.settings.deep_depth,
            default_multipv=self.settings.deep_multipv,
        )

        self.schedule_analysis()

    def set_state_notifier(self, notifier: Callable[[dict], None]) -> None:
        self._state_notifier = notifier

    def shutdown(self) -> None:
        self.engine_service.shutdown()

    def _snapshot_state(self) -> dict:
        with self._lock:
            return self.game.to_state_dict()

    def _emit_state_changed(self) -> None:
        if self._state_notifier is None:
            return
        self._state_notifier(self._snapshot_state())

    def _make_snapshot(
        self,
        *,
        depth: int | None = None,
        multipv: int | None = None,
    ) -> AnalysisSnapshot:
        with self._lock:
            return AnalysisSnapshot(
                fen=self.game.board.to_fen(),
                move_count=len(self.game.move_history),
                result=self.game.result,
                depth=depth or self.settings.deep_depth,
                multipv=multipv or self.settings.deep_multipv,
            )

    def _on_analysis_ready(self, snapshot: AnalysisSnapshot, lines: list[dict]) -> None:
        changed = False

        with self._lock:
            same_position = (
                len(self.game.move_history) == snapshot.move_count
                and self.game.board.to_fen() == snapshot.fen
            )

            if same_position:
                self.game.last_top_lines = lines
                changed = True

            self.game.analysis_pending = False
            changed = True or changed

        if changed:
            self._emit_state_changed()

    def schedule_analysis(
        self,
        *,
        depth: int | None = None,
        multipv: int | None = None,
    ) -> None:
        snapshot = self._make_snapshot(depth=depth, multipv=multipv)

        with self._lock:
            self.game.analysis_pending = True

        self.engine_service.schedule_analysis(snapshot, self._on_analysis_ready)

    def get_state(self) -> dict:
        with self._lock:
            need_schedule = (not self.game.last_top_lines) and (not self.game.analysis_pending)

        if need_schedule:
            self.schedule_analysis()

        return self._snapshot_state()

    def new_game(self, fen: str | None = None) -> dict:
        with self._lock:
            self.game = Game(fen or self.settings.start_fen)

        self.schedule_analysis()
        state = self._snapshot_state()
        self._emit_state_changed()
        return state

    def apply_uci(self, uci: str) -> dict:
        uci = uci.strip()

        with self._lock:
            if self.game.result != "*":
                raise ValueError("Game already finished")

            current_fen = self.game.board.to_fen()

        try:
            ch_board = chess.Board(current_fen)
            ch_move = chess.Move.from_uci(uci)
        except Exception:
            raise ValueError("Invalid UCI move")

        if ch_move not in ch_board.legal_moves:
            raise ValueError("Illegal move in this position")

        with self._lock:
            applied = self.game.apply_uci(uci)
            if applied is None:
                raise ValueError("Could not apply move")

        self.schedule_analysis()
        state = self._snapshot_state()
        self._emit_state_changed()
        return state

    def get_robot_best_move_payload(self) -> dict:
        with self._lock:
            top_lines = self.game.last_top_lines
            pending = self.game.analysis_pending

        payload = best_move_payload_from_top_lines(top_lines)

        return {
            "analysis_pending": pending,
            "available": payload is not None,
            "move": payload,
        }
