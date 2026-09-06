"""
vision/app/run_live.py — a tracker újraindítása a robot lépése után a backend
aktuális állásából (nem az AppConfig.start_fen-ből), és a moves.csv új
latencia-oszlopai.
"""
from __future__ import annotations

import csv
from pathlib import Path

import chess
import numpy as np
import pytest

from chess_logic import board_to_occupancy
from vision.app.config import AppConfig
from vision.app.run_live import LiveConfig, LiveProcessor

MID = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
WEIGHTS = Path(AppConfig().weights_path)


@pytest.fixture
def processor(monkeypatch):
    # nincs kamera / backend; a modell betöltését kiváltjuk, hogy ne kelljen checkpoint
    from vision.pipeline import tracker as tracker_mod
    monkeypatch.setattr(tracker_mod.ChessVisionTracker, "_load_model", lambda self: object())
    live = LiveConfig(backend_enabled=False, enable_timing=False, show_preview=False)
    return LiveProcessor(AppConfig(enable_pipeline_profiler=False), live)


def test_reset_tracker_uses_given_fen(processor):
    processor._reset_tracker(start_fen=MID)
    tr = processor.tracker
    assert tr.game.board.fen() == MID
    assert np.array_equal(tr.expected_start_occ, np.asarray(board_to_occupancy(chess.Board(MID))))
    # FEN nélkül vissza az AppConfig kezdőállására
    processor._reset_tracker()
    assert processor.tracker.game.board.fen() == chess.STARTING_FEN


def test_backend_current_fen_is_used_after_robot_move(processor, monkeypatch):
    class FakeBackend:
        def current_fen(self):
            return MID

    processor.backend = FakeBackend()
    # ugyanaz az út, mint a worker loopban a robot végeztével
    fen = processor.backend.current_fen()
    processor._reset_tracker(start_fen=fen)
    assert processor.tracker.game.board.fen() == MID


def test_moves_csv_has_latency_columns(processor, tmp_path):
    processor._move_records.append({
        "move_num": 1, "uci": "e2e4", "latency_ms": "1200.00", "latency_from_state_ms": "300.00",
        "latency_from_static_ms": "233.00", "frames_to_detect": 9, "mode": "exact",
    })
    out = tmp_path / "moves.csv"
    processor._save_moves_csv(out)
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    assert rows[0]["latency_from_state_ms"] == "300.00" and rows[0]["latency_from_static_ms"] == "233.00"
