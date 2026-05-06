from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from typing import Callable

import chess
import chess.engine

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AnalysisSnapshot:
    fen: str
    move_count: int
    result: str
    depth: int
    multipv: int


def _format_score(score_obj: chess.engine.PovScore) -> tuple[float, int | None]:
    mate_in = score_obj.mate()
    if mate_in is not None:
        cp_equiv = 1000.0 if mate_in > 0 else -1000.0
        return cp_equiv, mate_in

    score_cp = score_obj.score(mate_score=100000)
    if score_cp is None:
        return 0.0, None

    return score_cp / 100.0, None


class EngineAnalysisService:
    def __init__(self, engine_path: str, default_depth: int = 16, default_multipv: int = 3):
        self.default_depth = int(default_depth)
        self.default_multipv = int(default_multipv)

        self.engine = chess.engine.SimpleEngine.popen_uci([engine_path])
        self.engine_lock = threading.Lock()

        self.analysis_queue: queue.Queue = queue.Queue()
        self.analysis_thread = threading.Thread(target=self._analysis_worker, daemon=True)
        self.analysis_thread.start()

    def shutdown(self) -> None:
        self.analysis_queue.put(None)
        self.analysis_thread.join(timeout=2.0)

        with self.engine_lock:
            try:
                self.engine.quit()
            except Exception:
                pass

    def analyse_top_lines(
        self,
        ch_board: chess.Board,
        *,
        depth: int | None = None,
        multipv: int | None = None,
    ) -> list[dict]:
        depth = depth or self.default_depth
        multipv = multipv or self.default_multipv

        with self.engine_lock:
            infos = self.engine.analyse(
                ch_board,
                chess.engine.Limit(depth=depth),
                multipv=multipv,
            )

        if not isinstance(infos, list):
            infos = [infos]

        lines: list[dict] = []

        for info in infos:
            score_obj = info["score"].pov(chess.WHITE)
            score_cp, mate_in = _format_score(score_obj)

            pv_moves = info.get("pv") or []
            tmp_board = ch_board.copy()
            san_moves: list[str] = []

            for mv in pv_moves[:12]:
                san_moves.append(tmp_board.san(mv))
                tmp_board.push(mv)

            lines.append({
                "score": score_cp,
                "line_san": " ".join(san_moves),
                "pv_uci": [m.uci() for m in pv_moves],
                "mate_in": mate_in,
            })

        lines.sort(key=lambda l: l["score"], reverse=True)
        return lines[:multipv]

    def schedule_analysis(
        self,
        snapshot: AnalysisSnapshot,
        on_ready: Callable[[AnalysisSnapshot, list[dict]], None],
    ) -> None:
        self.analysis_queue.put((snapshot, on_ready))

    def _analysis_worker(self) -> None:
        while True:
            item = self.analysis_queue.get()
            if item is None:
                break

            snapshot, on_ready = item

            try:
                if snapshot.result != "*":
                    lines = [{
                        "score": 0.0,
                        "line_san": f"Game over: {snapshot.result}",
                        "pv_uci": [],
                        "mate_in": None,
                    }]
                else:
                    ch_board = chess.Board(snapshot.fen)
                    lines = self.analyse_top_lines(
                        ch_board,
                        depth=snapshot.depth,
                        multipv=snapshot.multipv,
                    )
            except Exception as exc:
                logger.error("Analysis failed: %s", exc)
                lines = []

            on_ready(snapshot, lines)
            self.analysis_queue.task_done()
