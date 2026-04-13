from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from chess_logic import StateStabilizer


DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parents[1] / "models" / "weights" / "resnet18_best_topdown.pt"


@dataclass
class AppConfig:
    start_fen: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    weights_path: Optional[str] = str(DEFAULT_WEIGHTS_PATH)

    inner_pad_ratio: float = 0.06
    context: float = 0.50
    cell: int = 96

    init_buffer_frames: int = 3
    init_max_dist: int = 2

    fuzzy_max_noise_cells: int = 1
    fuzzy_max_weighted_cost: float = 0.9

    enable_pipeline_profiler: bool = False

    partial_reclassify: bool = True
    partial_diff_threshold: float = 18.0
    partial_max_squares: int = 12

    full_reclassify_interval: int = 30

    @property
    def warp_size(self):
        s = 17 * self.cell
        return (s, s)


def make_stabilizer():
    return StateStabilizer(
        buffer_size=5,
        stable_frames=4,
        min_votes_ratio=0.65,
        emit_cooldown_s=0.20,
        min_mean_conf=0.50,
        min_changed_conf=0.50,
        max_changed_for_move=6,
        hold_changed_threshold=10,
        hold_low_conf_threshold=0.35,
        hold_min_duration_s=0.50,
        recovery_stable_frames=2,
    )