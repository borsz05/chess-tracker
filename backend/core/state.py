from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, Optional

import chess

from chess_logic import Game
from backend.services.engine_service import AnalysisSnapshot, EngineAnalysisService
from backend.services.move_payload import best_move_payload_from_top_lines


@dataclass(slots=True)
class BackendStateSettings:
    start_fen: str
    stockfish_path: str
    deep_depth: int
    deep_multipv: int
    robot_enabled: bool
    robot_color: str


class BackendState:
    def __init__(self, settings):
        self.settings = BackendStateSettings(
            start_fen=settings.start_fen,
            stockfish_path=settings.stockfish_path,
            deep_depth=settings.deep_depth,
            deep_multipv=settings.deep_multipv,
            robot_enabled=settings.robot_enabled,
            robot_color=settings.robot_color,
        )

        self._lock = threading.Lock()
        self._state_notifier: Optional[Callable[[dict], None]] = None

        self.game = Game(self.settings.start_fen)
        self.robot_busy: bool = False

        self.robot_service = None
        if self.settings.robot_enabled:
            self._try_init_robot()

        self.engine_service = EngineAnalysisService(
            engine_path=self.settings.stockfish_path,
            default_depth=self.settings.deep_depth,
            default_multipv=self.settings.deep_multipv,
        )

        self.schedule_analysis()

    def _try_init_robot(self) -> None:
        try:
            from backend.services.robot_exec_service import RobotExecutionService
            self.robot_service = RobotExecutionService()
            print("[Robot] Kapcsolódva, kész.", flush=True)
        except Exception as e:
            self.robot_service = None
            print(f"[Robot] Nem elérhető – vision-only módban fut. ({e})", flush=True)

    # ------------------------------------------------------------------
    # Notifier
    # ------------------------------------------------------------------

    def set_state_notifier(self, notifier: Callable[[dict], None]) -> None:
        self._state_notifier = notifier

    def shutdown(self) -> None:
        self.engine_service.shutdown()

    # ------------------------------------------------------------------
    # State snapshot
    # ------------------------------------------------------------------

    def _snapshot_state(self) -> dict:
        with self._lock:
            state = self.game.to_state_dict()
            state["robot_busy"] = self.robot_busy
            return state

    def _emit_state_changed(self) -> None:
        if self._state_notifier is None:
            return
        self._state_notifier(self._snapshot_state())

    # ------------------------------------------------------------------
    # Stockfish analysis
    # ------------------------------------------------------------------

    def _make_snapshot(
        self,
        *,
        depth: int | None = None,
        multipv: int | None = None,
    ) -> AnalysisSnapshot:
        with self._lock:
            return AnalysisSnapshot(
                fen=self.game.board.fen(),
                move_count=len(self.game.move_history),
                result=self.game.result,
                depth=depth or self.settings.deep_depth,
                multipv=multipv or self.settings.deep_multipv,
            )

    def _on_analysis_ready(self, snapshot: AnalysisSnapshot, lines: list[dict]) -> None:
        should_robot_play = False
        board_snap = None

        with self._lock:
            same_position = (
                len(self.game.move_history) == snapshot.move_count
                and self.game.board.fen() == snapshot.fen
            )
            if same_position:
                self.game.last_top_lines = lines

                # Meghatározzuk hogy a robot játsszon-e:
                # - van robot service
                # - a robot színe következik
                # - a játék még nem ért véget
                # - nincs már folyamatban robot mozgás
                robot_turn = (
                    (self.settings.robot_color == "black" and self.game.board.turn == chess.BLACK)
                    or
                    (self.settings.robot_color == "white" and self.game.board.turn == chess.WHITE)
                )
                should_robot_play = (
                    self.robot_service is not None
                    and robot_turn
                    and self.game.result == "*"
                    and not self.robot_busy
                    and bool(lines)
                )
                if should_robot_play:
                    # Board snapshot a lépés ELŐTTI állapotról (ez kell a build_move_descriptor-nak)
                    board_snap = self.game.board.copy()

            self.game.analysis_pending = False

        self._emit_state_changed()

        if should_robot_play:
            best_uci = (lines[0].get("pv_uci") or [None])[0]
            if best_uci:
                threading.Thread(
                    target=self._run_robot_move,
                    args=(best_uci, board_snap),
                    daemon=True,
                ).start()

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

    # ------------------------------------------------------------------
    # Robot execution (háttérszálban fut)
    # ------------------------------------------------------------------

    def _run_robot_move(self, uci: str, board_before: chess.Board) -> None:
        """Végrehajtja a robot lépését fizikailag, majd alkalmazza a játékban."""
        with self._lock:
            self.robot_busy = True
        self._emit_state_changed()
        print(f"[Robot] Lépés indítása: {uci}", flush=True)

        try:
            # Ez blokkolja a szálat amíg a robot fizikailag meg nem csinálja (~10-30 mp)
            self.robot_service.execute(uci, board_before)
            print(f"[Robot] Lépés kész: {uci}", flush=True)

            # Alkalmazza a lépést a backend játékállapotára
            with self._lock:
                applied = self.game.apply_uci(uci)

            if applied is not None:
                self.schedule_analysis()
                self._emit_state_changed()
            else:
                print(f"[Robot] Figyelmeztetés: {uci} illegális volt a backendben", flush=True)

        except Exception as e:
            print(f"[Robot] Hiba a lépés közben: {e}", flush=True)

        finally:
            with self._lock:
                self.robot_busy = False
            self._emit_state_changed()
            print("[Robot] Kész, vision folytatódhat.", flush=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        with self._lock:
            need_schedule = (not self.game.last_top_lines) and (not self.game.analysis_pending)

        if need_schedule:
            self.schedule_analysis()

        return self._snapshot_state()

    def new_game(self, fen: str | None = None) -> dict:
        with self._lock:
            self.game = Game(fen or self.settings.start_fen)

        if self.robot_service is not None:
            self.robot_service.reset_graveyard()

        self.schedule_analysis()
        state = self._snapshot_state()
        self._emit_state_changed()
        return state

    def apply_uci(self, uci: str) -> dict:
        uci = uci.strip()

        try:
            chess.Move.from_uci(uci)
        except Exception:
            raise ValueError("Invalid UCI move")

        with self._lock:
            if self.game.result != "*":
                raise ValueError("Game already finished")

            applied = self.game.apply_uci(uci)
            if applied is None:
                raise ValueError("Illegal move in this position")

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
