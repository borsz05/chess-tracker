from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from chess_logic import Board, Game, board_to_occupancy
from vision.app.config import AppConfig, make_stabilizer
from vision.pipeline.batch_classifier import classify_frame_batch, classify_selected_squares
from vision.pipeline.board_detector import detect_board_on_frame
from vision.pipeline.square_diff import compute_square_diffs


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
    camera_transform_name: str | None


def t_id(grid: np.ndarray) -> np.ndarray:
    return grid.copy()


def t_rot90_cw(grid: np.ndarray) -> np.ndarray:
    return np.rot90(grid, -1).copy()


def t_rot180(grid: np.ndarray) -> np.ndarray:
    return np.rot90(grid, 2).copy()


def t_rot90_ccw(grid: np.ndarray) -> np.ndarray:
    return np.rot90(grid, 1).copy()


def t_flip_lr(grid: np.ndarray) -> np.ndarray:
    return np.fliplr(grid).copy()


def transform_candidates():
    rotations = [
        ("id", t_id),
        ("rot90_cw", t_rot90_cw),
        ("rot180", t_rot180),
        ("rot90_ccw", t_rot90_ccw),
    ]

    candidates = []
    for name, fn in rotations:
        candidates.append((name, fn))
        candidates.append((f"{name}+flip_lr", lambda grid, fn=fn: t_flip_lr(fn(grid))))
    return candidates


def apply_transform(grid: np.ndarray, transform) -> np.ndarray:
    _, fn = transform
    return fn(grid)


def raw_to_standard(grid: np.ndarray, camera_transform) -> np.ndarray:
    # A nyers kamera-orientációból a belső standard orientációba forgatunk.
    camera_view = apply_transform(grid, camera_transform)
    return t_rot90_ccw(camera_view)


def choose_camera_transform_from_matrix(centers_img):
    best = None

    for transform in transform_candidates():
        pts = apply_transform(np.asarray(centers_img, dtype=np.float32), transform)

        dxs: list[float] = []
        dys: list[float] = []

        for row in range(8):
            for col in range(7):
                x0, y0 = pts[row][col]
                x1, y1 = pts[row][col + 1]
                dxs.append(float(x1 - x0))

        for row in range(7):
            for col in range(8):
                x0, y0 = pts[row][col]
                x1, y1 = pts[row + 1][col]
                dys.append(float(y1 - y0))

        mean_dx = sum(dxs) / len(dxs)
        mean_dy = sum(dys) / len(dys)
        pos_dx_ratio = sum(delta > 0 for delta in dxs) / len(dxs)
        pos_dy_ratio = sum(delta > 0 for delta in dys) / len(dys)

        score = (pos_dx_ratio + pos_dy_ratio, mean_dx + mean_dy)
        if best is None or score > best[0]:
            best = (score, transform)

    _, transform = best
    return transform


def occ_distance(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def mean_conf(confs: np.ndarray) -> float:
    return float(np.mean(confs))


def disturbance_score(prev_occ: np.ndarray | None, curr_occ: np.ndarray | None) -> int:
    if prev_occ is None or curr_occ is None:
        return 0
    return occ_distance(prev_occ, curr_occ)


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


def most_common_move(candidate_buf):
    if not candidate_buf:
        return None, 0

    counts: dict[str, int] = {}
    for move in candidate_buf:
        counts[move] = counts.get(move, 0) + 1

    best_uci = max(counts, key=counts.get)
    return best_uci, counts[best_uci]


class ChessVisionTracker:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

        self.game = Game(cfg.start_fen)
        self.stabilizer = make_stabilizer()
        self.occ_model = self._load_model()

        self.expected_start_occ = np.asarray(
            board_to_occupancy(Board(cfg.start_fen)),
            dtype=np.int32,
        )

        self.det = None
        self.camera_transform = None
        self.camera_transform_name = None

        self.accepted_occ = None
        self.observed_occ = None
        self.last_raw_occ = None

        self.init_label_grids: list[np.ndarray] = []
        self.init_conf_grids: list[np.ndarray] = []
        self.init_sample_counter = 0
        self.initialized = False

        self.prev_warp = None
        self.prev_raw_labels = None
        self.prev_raw_confs = None
        self.frame_counter = 0

        self.candidate_move_buf = deque(maxlen=cfg.move_vote_window)

    def _create_profiler(self):
        if not self.cfg.enable_pipeline_profiler:
            return None

        from vision.pipeline.profiler import PipelineProfiler

        return PipelineProfiler()

    def _default_weights_path(self) -> str:
        return str(
            Path(__file__).resolve().parents[1]
            / "models"
            / "weights"
            / "resnet18_best_szines_topdown_kepeken.pt"
        )

    def _load_model(self):
        profiler = self._create_profiler()
        if profiler:
            profiler.start("model_load")

        from vision.models.occupancy_color_model import OccupancyColorModel

        weights_path = self.cfg.weights_path or self._default_weights_path()
        model = OccupancyColorModel(weights_path=weights_path)

        if profiler:
            profiler.stop("model_load")

        self.profiler = profiler
        return model

    def _profile_start(self, name: str):
        if self.profiler:
            self.profiler.start(name)

    def _profile_stop(self, name: str):
        if self.profiler:
            self.profiler.stop(name)

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
            camera_transform_name=self.camera_transform_name,
        )

    def _full_classify(self, frame_bgr: np.ndarray):
        self._profile_start("classifier_full")
        cls = classify_frame_batch(
            frame_bgr,
            self.det,
            self.cfg.warp_size,
            self.occ_model,
            context=self.cfg.context,
        )
        self._profile_stop("classifier_full")
        return cls

    def _classify_selected_squares(self, img_warp: np.ndarray, changed_squares):
        self._profile_start("classifier_partial")
        updates = classify_selected_squares(
            img_warp,
            self.det.bbox_warp,
            changed_squares,
            self.occ_model,
            context=self.cfg.context,
        )
        self._profile_stop("classifier_partial")
        return updates

    def _choose_changed_squares(self, img_warp: np.ndarray):
        self._profile_start("square_diff")
        diffs = compute_square_diffs(self.prev_warp, img_warp, self.det.bbox_warp)
        self._profile_stop("square_diff")

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
        class Result:
            pass

        result = Result()
        result.labels = self.prev_raw_labels.copy()
        result.confs = self.prev_raw_confs.copy()
        return result

    def _partial_or_full_classify(self, frame_bgr: np.ndarray, img_warp: np.ndarray):
        if not self._should_use_partial_reclassify():
            return self._full_classify(frame_bgr)

        changed_squares = self._choose_changed_squares(img_warp)
        if not changed_squares:
            return self._copy_previous_classification()

        updates = self._classify_selected_squares(img_warp, changed_squares)

        class Result:
            pass

        result = Result()
        result.labels = self.prev_raw_labels.copy()
        result.confs = self.prev_raw_confs.copy()

        for (row, col), (label, conf) in updates.items():
            result.labels[row, col] = label
            result.confs[row, col] = conf

        return result

    def _detect_board(self, gray: np.ndarray):
        self._profile_start("board_detect")
        det = detect_board_on_frame(
            gray,
            cell=self.cfg.cell,
            inner_pad_ratio=self.cfg.inner_pad_ratio,
        )
        self._profile_stop("board_detect")
        return det

    def _reset_init_buffers(self):
        self.det = None
        self.init_label_grids.clear()
        self.init_conf_grids.clear()
        self.init_sample_counter = 0

    def _append_init_sample(self, cls):
        self.init_sample_counter += 1
        should_store = len(self.init_label_grids) == 0 or (self.init_sample_counter % self.cfg.init_sample_every) == 0
        if should_store:
            self.init_label_grids.append(cls.labels)
            self.init_conf_grids.append(cls.confs)

    def _init_samples_ready(self) -> bool:
        return len(self.init_label_grids) >= self.cfg.init_buffer_frames

    def _choose_camera_transform(self, cls):
        if self.camera_transform is not None:
            return None

        if getattr(self.det, "centers_img", None) is None:
            self._reset_init_buffers()
            return self._make_result(
                initialized=False,
                board_changed=False,
                raw_labels=cls.labels,
                confs_std=None,
                san=None,
                uci=None,
                mode="init-no-centers-img",
                raw_dist=0,
                obs_mean=0.0,
            )

        self.camera_transform = choose_camera_transform_from_matrix(self.det.centers_img)
        self.camera_transform_name, _ = self.camera_transform
        return None

    def _vote_init_grids(self):
        init_labels_std = [raw_to_standard(grid, self.camera_transform) for grid in self.init_label_grids]
        init_confs_std = [raw_to_standard(grid, self.camera_transform) for grid in self.init_conf_grids]
        return weighted_vote_occ(init_labels_std, init_confs_std)

    def _store_init_baseline(self, frame_bgr: np.ndarray, cls, init_labels, init_confs):
        self.accepted_occ = init_labels.copy()
        self.observed_occ = init_labels.copy()
        self.last_raw_occ = init_labels.copy()
        self.stabilizer.update(init_labels.tolist(), init_confs.tolist())
        self.initialized = True

        # Ezek kellenek a későbbi partial reclassify útvonalhoz.
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
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if self.det is None:
            self.det = self._detect_board(gray)
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

        transform_error = self._choose_camera_transform(cls)
        if transform_error is not None:
            return transform_error

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
        confs_std = raw_to_standard(cls.confs, self.camera_transform)

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
        self._profile_start("warp")
        img_warp = cv2.warpPerspective(
            frame_bgr,
            self.det.M,
            self.cfg.warp_size,
            flags=cv2.WARP_INVERSE_MAP,
        )
        self._profile_stop("warp")
        return img_warp

    def _update_frame_cache(self, img_warp: np.ndarray, cls):
        self.prev_warp = img_warp.copy()
        self.prev_raw_labels = cls.labels.copy()
        self.prev_raw_confs = cls.confs.copy()
        self.frame_counter += 1

    def _stabilize_observation(self, labels_std: np.ndarray, confs_std: np.ndarray):
        self._profile_start("stabilizer")
        decision = self.stabilizer.update(labels_std.tolist(), confs_std.tolist())
        self._profile_stop("stabilizer")
        return decision

    def _resolve_move(self, stable_occ: np.ndarray, confs_std: np.ndarray):
        self._profile_start("resolve")
        resolve_result = self.game.resolve_from_occupancy(
            stable_occ.tolist(),
            confs_std.tolist(),
            max_noise_cells=self.cfg.fuzzy_max_noise_cells,
            max_weighted_cost=self.cfg.fuzzy_max_weighted_cost,
        )
        self._profile_stop("resolve")
        return resolve_result

    def _apply_move(self, best_uci: str):
        self._profile_start("apply_move")
        applied_move = self.game.apply_uci(best_uci)
        self._profile_stop("apply_move")
        return applied_move

    def process_frame(self, frame_bgr: np.ndarray) -> FrameProcessResult:
        if not self.initialized:
            return self.try_initialize_from_frame(frame_bgr)

        img_warp = self._warp_frame(frame_bgr)
        cls = self._partial_or_full_classify(frame_bgr, img_warp)

        raw_labels = cls.labels
        labels_std = raw_to_standard(cls.labels, self.camera_transform)
        confs_std = raw_to_standard(cls.confs, self.camera_transform)

        self.observed_occ = labels_std
        obs_mean = mean_conf(confs_std)

        raw_dist = disturbance_score(self.last_raw_occ, labels_std)
        self.last_raw_occ = labels_std

        self._update_frame_cache(img_warp, cls)

        decision = self._stabilize_observation(labels_std, confs_std)
        if decision.emit_occ is None:
            return self._make_result(
                initialized=True,
                board_changed=False,
                raw_labels=raw_labels,
                confs_std=confs_std,
                san=None,
                uci=None,
                mode=f"{decision.mode}:{decision.reason}",
                raw_dist=raw_dist,
                obs_mean=obs_mean,
            )

        stable_occ = np.asarray(decision.emit_occ, dtype=np.int32)
        if occ_distance(self.accepted_occ, stable_occ) == 0:
            self.candidate_move_buf.clear()
            return self._make_result(
                initialized=True,
                board_changed=False,
                raw_labels=raw_labels,
                confs_std=confs_std,
                san=None,
                uci=None,
                mode="stable-same",
                raw_dist=raw_dist,
                obs_mean=obs_mean,
            )

        resolve_result = self._resolve_move(stable_occ, confs_std)
        if resolve_result.move is None:
            self.candidate_move_buf.clear()
            return self._make_result(
                initialized=True,
                board_changed=False,
                raw_labels=raw_labels,
                confs_std=confs_std,
                san=None,
                uci=None,
                mode=resolve_result.mode or "no-legal-fit",
                raw_dist=raw_dist,
                obs_mean=obs_mean,
            )

        uci = resolve_result.move.to_uci()
        self.candidate_move_buf.append(uci)
        best_uci, best_count = most_common_move(self.candidate_move_buf)

        if best_uci != uci or best_count < self.cfg.move_vote_min_count:
            return self._make_result(
                initialized=True,
                board_changed=False,
                raw_labels=raw_labels,
                confs_std=confs_std,
                san=None,
                uci=None,
                mode=f"move-vote {best_count}/{self.cfg.move_vote_min_count}",
                raw_dist=raw_dist,
                obs_mean=obs_mean,
            )

        applied_move = self._apply_move(best_uci)
        if applied_move is None:
            self.candidate_move_buf.clear()
            return self._make_result(
                initialized=True,
                board_changed=False,
                raw_labels=raw_labels,
                confs_std=confs_std,
                san=None,
                uci=None,
                mode="illegal-reject",
                raw_dist=raw_dist,
                obs_mean=obs_mean,
            )

        san = self.game.move_history[-1].san if self.game.move_history else None
        self.accepted_occ = np.asarray(resolve_result.expected_occ, dtype=np.int32)
        self.candidate_move_buf.clear()

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
