from __future__ import annotations

import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from chess_logic import Board, Game, board_to_occupancy
from vision.app.config import AppConfig, make_stabilizer
from vision.pipeline.batch_classifier import BatchClassificationResult, classify_frame_batch, classify_selected_squares
from vision.pipeline.board_detector import detect_board_on_frame
from vision.pipeline.profiler import PipelineProfiler


@dataclass
class FrameProcessResult:
    initialized: bool
    board_changed: bool
    accepted_occ: np.ndarray | None
    observed_occ: np.ndarray | None
    raw_labels: np.ndarray | None
    confs_std: np.ndarray | None
    san: str | None
    uci: str | None
    mode: str
    raw_dist: int
    obs_mean: float


def raw_to_standard(grid: np.ndarray) -> np.ndarray:
    # Camera always: top-left = A1, bottom-right = H8
    grid = np.rot90(grid, 1).copy()
    return np.fliplr(grid).copy()


def occ_distance(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


# Stabilizer reasons that mean "the pipeline agrees with itself right now".
# Anything else is the pipeline actively failing to converge.
_SETTLED_REASONS = frozenset({"no_change", "cooldown"})


def disturbance_score(prev_occ: np.ndarray | None, curr_occ: np.ndarray | None) -> int:
    if prev_occ is None or curr_occ is None:
        return 0
    return occ_distance(prev_occ, curr_occ)


def compute_square_diffs(prev_warp, curr_warp, bbox_grid):
    diffs = []
    for r in range(8):
        for c in range(8):
            x0, y0, x1, y1 = bbox_grid[r][c]
            prev_sq = prev_warp[y0:y1, x0:x1]
            curr_sq = curr_warp[y0:y1, x0:x1]
            if prev_sq.size == 0 or curr_sq.size == 0:
                diff = 0
            else:
                d = np.abs(prev_sq.astype(np.int16) - curr_sq.astype(np.int16))
                diff = float(np.mean(d))
            diffs.append((diff, r, c))
    diffs.sort(reverse=True)
    return diffs


def weighted_vote_occ(label_grids: Iterable[np.ndarray], conf_grids: Iterable[np.ndarray]):
    label_stack = np.stack(list(label_grids), axis=0).astype(np.int32)
    conf_stack = np.stack(list(conf_grids), axis=0).astype(np.float32)

    weights = np.zeros((3, 8, 8), dtype=np.float32)
    for cls in range(3):
        weights[cls] = np.sum(np.where(label_stack == cls, conf_stack, 0.0), axis=0)

    best_labels = np.argmax(weights, axis=0).astype(np.int32)
    best_weights = np.take_along_axis(weights, best_labels[None, :, :], axis=0)[0]
    total_weights = np.sum(weights, axis=0)
    best_confs = np.divide(best_weights, np.maximum(total_weights, 1e-9)).astype(np.float32)

    return best_labels, best_confs


class ChessVisionTracker:
    def __init__(self, cfg: AppConfig, model=None):
        self.cfg = cfg

        self.game = Game(cfg.start_fen)
        self.stabilizer = make_stabilizer()

        self.profiler: PipelineProfiler | None = (
            PipelineProfiler() if cfg.enable_pipeline_profiler else None
        )

        self.occ_model = model if model is not None else self._load_model()

        self.expected_start_occ = np.asarray(
            board_to_occupancy(Board(cfg.start_fen)),
            dtype=np.int32,
        )

        self.det = None

        self.accepted_occ = None
        self.observed_occ = None
        self.last_std_occ = None

        self.init_label_grids: list[np.ndarray] = []
        self.init_conf_grids: list[np.ndarray] = []
        self.initialized = False

        self.prev_warp = None
        self.prev_raw_labels = None
        self.prev_raw_confs = None
        self.frame_counter = 0

        # Board re-detection on sustained stall
        self._unsettled_since: float | None = None
        self._last_redetect_t: float = 0.0

        # Init phase timing (perf_counter-based, ms)
        self._init_start: float | None = None
        self._init_frames: int = 0
        self._init_board_detect_ms: float | None = None
        self._init_total_ms: float | None = None

    def _load_model(self):
        if self.profiler:
            self.profiler.start("model_load")
        from vision.models.occupancy_color_model import OccupancyColorModel
        model = OccupancyColorModel(weights_path=self.cfg.weights_path)
        if self.profiler:
            self.profiler.stop("model_load")
        return model

    @property
    def init_stats(self) -> dict | None:
        """Returns init-phase timing info once initialized, else None."""
        if not self.initialized:
            return None
        return {
            "frames": self._init_frames,
            "board_detect_ms": self._init_board_detect_ms,
            "total_ms": self._init_total_ms,
        }

    @contextmanager
    def _profile(self, name: str):
        if self.profiler:
            self.profiler.start(name)
        try:
            yield
        finally:
            if self.profiler:
                self.profiler.stop(name)

    def _no_move_result(self, raw_labels, confs_std, mode: str, raw_dist: int, obs_mean: float) -> FrameProcessResult:
        return self._make_result(
            initialized=True,
            board_changed=False,
            raw_labels=raw_labels,
            confs_std=confs_std,
            san=None,
            uci=None,
            mode=mode,
            raw_dist=raw_dist,
            obs_mean=obs_mean,
        )

    def _make_result(
        self,
        *,
        initialized: bool,
        board_changed: bool,
        raw_labels,
        confs_std,
        san,
        uci,
        mode: str,
        raw_dist: int,
        obs_mean: float,
    ) -> FrameProcessResult:
        accepted_occ = None if self.accepted_occ is None else self.accepted_occ.copy()
        observed_occ = None if self.observed_occ is None else self.observed_occ.copy()

        return FrameProcessResult(
            initialized=initialized,
            board_changed=board_changed,
            accepted_occ=accepted_occ,
            observed_occ=observed_occ,
            raw_labels=raw_labels,
            confs_std=confs_std,
            san=san,
            uci=uci,
            mode=mode,
            raw_dist=raw_dist,
            obs_mean=obs_mean,
        )

    def _full_classify(self, frame_bgr: np.ndarray):
        with self._profile("classifier_full"):
            return classify_frame_batch(
                frame_bgr,
                self.det,
                self.cfg.warp_size,
                self.occ_model,
                context=self.cfg.context,
            )

    def _classify_selected_squares(self, img_warp: np.ndarray, changed_squares):
        with self._profile("classifier_partial"):
            return classify_selected_squares(
                img_warp,
                self.det.bbox_warp,
                changed_squares,
                self.occ_model,
                context=self.cfg.context,
            )

    def _choose_changed_squares(self, img_warp: np.ndarray):
        with self._profile("square_diff"):
            diffs = compute_square_diffs(self.prev_warp, img_warp, self.det.bbox_warp)
        return [
            (row, col)
            for diff, row, col in diffs[: self.cfg.partial_max_squares]
            if diff > self.cfg.partial_diff_threshold
        ]

    def _should_use_partial_reclassify(self) -> bool:
        return (
            self.cfg.partial_reclassify
            and self.prev_warp is not None
            and self.prev_raw_labels is not None
            and self.prev_raw_confs is not None
            and self.frame_counter % self.cfg.full_reclassify_interval != 0
        )

    def _copy_previous_classification(self):
        return BatchClassificationResult(
            labels=self.prev_raw_labels.copy(),
            confs=self.prev_raw_confs.copy(),
        )

    def _partial_or_full_classify(self, frame_bgr: np.ndarray, img_warp: np.ndarray):
        if not self._should_use_partial_reclassify():
            return self._full_classify(frame_bgr)

        changed_squares = self._choose_changed_squares(img_warp)
        if not changed_squares:
            return self._copy_previous_classification()

        updates = self._classify_selected_squares(img_warp, changed_squares)

        labels = self.prev_raw_labels.copy()
        confs = self.prev_raw_confs.copy()

        for (row, col), (label, conf) in updates.items():
            labels[row, col] = label
            confs[row, col] = conf

        return BatchClassificationResult(labels=labels, confs=confs)

    def _detect_board(self, gray: np.ndarray):
        with self._profile("board_detect"):
            return detect_board_on_frame(
                gray,
                cell=self.cfg.cell,
                inner_pad_ratio=self.cfg.inner_pad_ratio,
            )

    def _reset_init_buffers(self):
        self.det = None
        self.init_label_grids.clear()
        self.init_conf_grids.clear()

    def _append_init_sample(self, cls):
        self.init_label_grids.append(cls.labels)
        self.init_conf_grids.append(cls.confs)

    def _init_samples_ready(self) -> bool:
        return len(self.init_label_grids) >= self.cfg.init_buffer_frames

    def _vote_init_grids(self):
        init_labels_std = [raw_to_standard(grid) for grid in self.init_label_grids]
        init_confs_std = [raw_to_standard(grid) for grid in self.init_conf_grids]
        return weighted_vote_occ(init_labels_std, init_confs_std)

    def _store_init_baseline(self, frame_bgr: np.ndarray, cls, init_labels, init_confs):
        self.accepted_occ = init_labels.copy()
        self.observed_occ = init_labels.copy()
        self.last_std_occ = init_labels.copy()
        self.stabilizer.update(init_labels.tolist(), init_confs.tolist())
        self.initialized = True

        img_warp = cv2.warpPerspective(
            frame_bgr,
            self.det.M,
            self.cfg.warp_size,
            flags=cv2.WARP_INVERSE_MAP,
        )
        self.prev_warp = img_warp.copy()
        self.prev_raw_labels = cls.labels.copy()
        self.prev_raw_confs = cls.confs.copy()
        self.frame_counter = 1

    def try_initialize_from_frame(self, frame_bgr: np.ndarray) -> FrameProcessResult:
        if self._init_start is None:
            self._init_start = time.perf_counter()
        self._init_frames += 1

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if self.det is None:
            t_det = time.perf_counter()
            self.det = self._detect_board(gray)
            if self.det.ok and self._init_board_detect_ms is None:
                self._init_board_detect_ms = (time.perf_counter() - t_det) * 1000.0
            if not self.det.ok:
                self.det = None
                return self._make_result(
                    initialized=False,
                    board_changed=False,
                    raw_labels=None,
                    confs_std=None,
                    san=None,
                    uci=None,
                    mode="detect-failed",
                    raw_dist=0,
                    obs_mean=0.0,
                )

        cls = self._full_classify(frame_bgr)
        self._append_init_sample(cls)

        if not self._init_samples_ready():
            return self._make_result(
                initialized=False,
                board_changed=False,
                raw_labels=cls.labels,
                confs_std=None,
                san=None,
                uci=None,
                mode=f"init-buffer {len(self.init_label_grids)}/{self.cfg.init_buffer_frames}",
                raw_dist=0,
                obs_mean=0.0,
            )

        init_labels, init_confs = self._vote_init_grids()
        init_dist = occ_distance(init_labels, self.expected_start_occ)

        if init_dist > self.cfg.init_max_dist:
            self._reset_init_buffers()
            return self._make_result(
                initialized=False,
                board_changed=False,
                raw_labels=cls.labels,
                confs_std=None,
                san=None,
                uci=None,
                mode=f"init-too-far dist={init_dist}",
                raw_dist=0,
                obs_mean=0.0,
            )

        self._store_init_baseline(frame_bgr, cls, init_labels, init_confs)
        confs_std = raw_to_standard(cls.confs)

        if self._init_start is not None:
            self._init_total_ms = (time.perf_counter() - self._init_start) * 1000.0
            if self.profiler:
                self.profiler.record("init_total", self._init_total_ms)
                if self._init_board_detect_ms is not None:
                    self.profiler.record("init_board_detect_ok", self._init_board_detect_ms)

        return self._make_result(
            initialized=True,
            board_changed=False,
            raw_labels=cls.labels,
            confs_std=confs_std,
            san=None,
            uci=None,
            mode="init-ok",
            raw_dist=0,
            obs_mean=float(np.mean(init_confs)),
        )

    def _warp_frame(self, frame_bgr: np.ndarray):
        with self._profile("warp"):
            return cv2.warpPerspective(
                frame_bgr,
                self.det.M,
                self.cfg.warp_size,
                flags=cv2.WARP_INVERSE_MAP,
            )

    def _update_frame_cache(self, img_warp: np.ndarray, cls):
        self.prev_warp = img_warp.copy()
        self.prev_raw_labels = cls.labels.copy()
        self.prev_raw_confs = cls.confs.copy()
        self.frame_counter += 1

    def _should_redetect_board(self, decision, now: float) -> bool:
        """True once the stabilizer has been unable to settle for long enough
        that the frozen homography is worth re-solving."""
        settled = decision.emit_occ is not None or decision.reason in _SETTLED_REASONS
        if settled:
            self._unsettled_since = None
            return False

        if self._unsettled_since is None:
            self._unsettled_since = now
            return False

        if (now - self._unsettled_since) < self.cfg.redetect_after_stuck_s:
            return False

        return (now - self._last_redetect_t) >= self.cfg.redetect_min_interval_s

    def _redetect_board(self, frame_bgr: np.ndarray) -> bool:
        """Re-solves the board homography and invalidates the frame cache."""
        self._last_redetect_t = time.time()
        self._unsettled_since = None

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        det = self._detect_board(gray)
        if not det.ok:
            return False

        self.det = det
        # The warp moved, so the cached previous frame and its labels no longer
        # line up with the new bboxes — drop them to force a full reclassify.
        self.prev_warp = None
        self.prev_raw_labels = None
        self.prev_raw_confs = None
        return True

    def _stabilize_observation(self, labels_std: np.ndarray, confs_std: np.ndarray):
        with self._profile("stabilizer"):
            return self.stabilizer.update(labels_std.tolist(), confs_std.tolist())

    def _resolve_move(self, stable_occ: np.ndarray, confs_std: np.ndarray):
        with self._profile("resolve"):
            return self.game.resolve_from_occupancy(
                stable_occ.tolist(),
                confs_std.tolist(),
                max_noise_cells=self.cfg.fuzzy_max_noise_cells,
                max_weighted_cost=self.cfg.fuzzy_max_weighted_cost,
            )

    def _apply_move(self, best_uci: str):
        with self._profile("apply_move"):
            return self.game.apply_uci(best_uci)

    def process_frame(self, frame_bgr: np.ndarray) -> FrameProcessResult:
        if not self.initialized:
            return self.try_initialize_from_frame(frame_bgr)

        img_warp = self._warp_frame(frame_bgr)
        cls = self._partial_or_full_classify(frame_bgr, img_warp)

        raw_labels = cls.labels
        labels_std = raw_to_standard(cls.labels)
        confs_std = raw_to_standard(cls.confs)

        self.observed_occ = labels_std
        obs_mean = float(np.mean(confs_std))

        raw_dist = disturbance_score(self.last_std_occ, labels_std)
        self.last_std_occ = labels_std

        self._update_frame_cache(img_warp, cls)

        decision = self._stabilize_observation(labels_std, confs_std)
        if decision.emit_occ is None:
            if self._should_redetect_board(decision, time.time()):
                ok = self._redetect_board(frame_bgr)
                mode = "redetect-ok" if ok else "redetect-failed"
                return self._no_move_result(raw_labels, confs_std, mode, raw_dist, obs_mean)
            return self._no_move_result(raw_labels, confs_std, f"{decision.mode}:{decision.reason}", raw_dist, obs_mean)

        stable_occ = np.asarray(decision.emit_occ, dtype=np.int32)
        if occ_distance(self.accepted_occ, stable_occ) == 0:
            return self._no_move_result(raw_labels, confs_std, "stable-same", raw_dist, obs_mean)

        resolve_result = self._resolve_move(stable_occ, confs_std)
        if resolve_result.move is None:
            return self._no_move_result(raw_labels, confs_std, resolve_result.mode or "no-legal-fit", raw_dist, obs_mean)

        best_uci = resolve_result.move.to_uci()

        applied_move = self._apply_move(best_uci)
        if applied_move is None:
            return self._no_move_result(raw_labels, confs_std, "illegal-reject", raw_dist, obs_mean)

        san = self.game.move_history[-1].san if self.game.move_history else None
        self.accepted_occ = np.asarray(resolve_result.expected_occ, dtype=np.int32)

        return self._make_result(
            initialized=True,
            board_changed=True,
            raw_labels=raw_labels,
            confs_std=confs_std,
            san=san,
            uci=best_uci,
            mode=resolve_result.mode or "exact",
            raw_dist=raw_dist,
            obs_mean=obs_mean,
        )