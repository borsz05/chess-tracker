from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vision.pipeline.stabilizer import StateStabilizer


DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parents[1] / "models" / "weights" / "resnet18_best_topdown.pt"


@dataclass
class AppConfig:
    start_fen: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    weights_path: str | None = str(DEFAULT_WEIGHTS_PATH)

    # Inferencia backend: "onnx" (alapértelmezett, ~8x gyorsabb CPU-n), "torch"
    # (fallback), "auto" (onnx ha van .onnx a .pt mellett, különben torch).
    # Az ONNX fájl: python -m tools.export_onnx --weights <weights_path>
    # (alapból <weights_path>.onnx-ra ír; onnx_path=None ezt keresi).
    inference_backend: str = "auto"
    onnx_path: str | None = None
    inference_threads: int | None = None
    # Ha az export (--int8) egy INT8 változatot ajánlott (black recall nem
    # romlott), auto módban azt töltjük; False -> mindig a fp32 .onnx.
    allow_int8: bool = True

    inner_pad_ratio: float = 0.06
    context: float = 0.50
    cell: int = 96

    init_buffer_frames: int = 3
    init_max_dist: int = 2

    fuzzy_max_noise_cells: int = 1
    fuzzy_max_weighted_cost: float = 0.9

    enable_pipeline_profiler: bool = True

    partial_reclassify: bool = True
    partial_diff_threshold: float = 18.0
    # A hand over the board dirties well over 12 squares; the ones that don't
    # fit kept stale labels and poisoned the vote. ~3.3 ms/square, so 20 is
    # still far inside the per-frame budget.
    partial_max_squares: int = 20

    full_reclassify_interval: int = 45

    # The homography is solved once at init and then frozen, so a nudged board
    # or camera degrades every later classification with no way back. When the
    # stabilizer cannot settle for this long, re-solve it (~1.4 s, hence the
    # cooldown between attempts).
    redetect_after_stuck_s: float = 4.0
    redetect_min_interval_s: float = 10.0

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
        # A one-cell difference is classifier noise, never a legal move, so it
        # no longer resets the persistence run — this is the main fix for
        # getting stuck in candidate_not_persistent.
        flicker_tolerance_cells=1,
        max_candidate_misses=3,
    )
