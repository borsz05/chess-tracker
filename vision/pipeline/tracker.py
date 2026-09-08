"""
ChessVisionTracker — a fő állapotgép (10. szakasz, átszervezett változat).

Képkockánként:
    warp -> (négyzetenkénti |diff| = mozgás-jel + változott mezők)
         -> klasszifikáció: teljes / részleges (változott + gördülő frissítés)
         -> raw_to_standard (FIX orientáció-leképezés — NEM változott)
         -> StateStabilizer.update(labels, confs, motion)   # mikor stabil
         -> Game.resolve_from_occupancy (+ típus-tipp promóciónál)
         -> elfogadás (előtag-kétértelműség / promóció-várakozás után)

Mi változott a régi trackerhez képest (docs/refaktor_prompt.md 10. szakasz):
  * nincs frame-throttle a trackerben; a hívó minden frame-et átadhat, a
    költség 15-35 ms (mért, docs/pipeline_tuning.md);
  * a mozgás-jel (a részleges újraklasszifikálás square_diff-je) a stabilizer
    mozgás-kapujába megy: amíg kéz/kar mozog a táblán, nincs kiadás;
  * gördülő frissítés: minden frame-ben +N mezőt újraosztályozunk körbejárva,
    így egy rossz címke nem marad meg a következő teljes átosztályozásig;
  * eseményvezérelt teljes átosztályozás, ha a stabilizer feloldhatatlan
    (1 mezős vagy legális lépés nélküli) eltérést lát;
  * a resolver expected_occ-ja a stabilizer REFERENCIÁJA lesz (set_reference),
    így "no_change" = a kamera azt látja, amit a sakklogika hisz;
  * promóció: a típus-fej választja a bábut; ha még gyalogot lát a célmezőn,
    a cseréig (max promotion_wait_s) várunk;
  * bástyával kezdett sánc (Rf1 ~ O-O előtagja): hosszabb megerősítés;
  * újradetektáláskor (és init-kor) a detektor rácsának orientációját az
    elfogadott/várt álláshoz igazítjuk a bbox-rács átrendezésével — a
    raw_to_standard leképezés érintetlen (vision/pipeline/orientation.py);
  * a tábladetektálás ~165 ms (vektorizált), újradetektálás után a cache ürül.

Idő: minden időzítés a `clock` hívásából jön (alapból time.time), így a
visszajátszó (tools/replay_frames.py) determinisztikus időbélyegekkel tud
futtatni.
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterable

import cv2
import numpy as np

from chess_logic import Board, Game, board_to_occupancy
from chess_logic.resolver import TYPE_INDEX
from vision.app.config import AppConfig, make_stabilizer
from vision.pipeline.batch_classifier import (
    BatchClassificationResult,
    classify_frame_batch,
    classify_squares_batch,
    classify_warp_squares_batch,
)
from vision.pipeline.board_detector import CenterGrid, DetectionResult, detect_board_on_frame
from vision.pipeline.orientation import IDENTITY, SYMMETRY_NAMES, apply_symmetry, choose_alignment, inverse_symmetry
from vision.pipeline.profiler import PipelineProfiler
from vision.pipeline.stabilizer import SETTLED_REASONS, StabilizerDecision


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
    # --- új diagnosztika (10. szakasz) ---
    reason: str = ""                       # a stabilizer indoklása / a tracker döntése
    motion: float | None = None            # a tábla mozgás-jele ebben a frame-ben
    accept_info: dict | None = None        # elfogadáskor: időbélyegek a latencia-méréshez


def raw_to_standard(grid: np.ndarray) -> np.ndarray:
    # Camera always: top-left = A1, bottom-right = H8
    grid = np.rot90(grid, 1).copy()
    return np.fliplr(grid).copy()


def standard_to_raw(grid: np.ndarray) -> np.ndarray:
    """A raw_to_standard inverze (fliplr, majd rot90 visszafelé)."""
    return np.rot90(np.fliplr(np.asarray(grid)), -1).copy()


def occ_distance(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


# A détektor rácsának lehetséges orientációi az igazításnál: csak FORGATÁSOK.
# A homográfia a kontúr konzisztens körüljárási irányából jön, tükrözést nem
# ad (a mért 18 elforgatott detektálás mind tiszta rot270 volt); a tükrözések
# kizárása teszi az alapállást (bal-jobb szimmetrikus) is egyértelművé.
ROTATION_SYMMETRIES = tuple(n for n in SYMMETRY_NAMES if not n.endswith("+flip"))


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


def align_detection_to_expected(
    det: DetectionResult,
    observed_raw: np.ndarray,
    expected_std: np.ndarray,
    *,
    max_mismatch: int,
    min_margin: int,
) -> tuple[str, list[tuple[str, int]]]:
    """
    Ha a detektor rácsa az elvárt álláshoz képest elforgatva áll, a det
    bbox/centers rácsait HELYBEN átrendezi úgy, hogy a későbbi nyers címkék
    raw_to_standard után az elvárt állást adják. Visszaad: (a detektor rácsán
    talált szimmetria, rangsor). IDENTITY = nem kellett igazítani.

    Biztonság: csak forgatások; csak egyértelmű illeszkedésnél (lásd
    orientation.choose_alignment). Tükrözés vagy kétes eset -> identitás.
    """
    expected_raw = standard_to_raw(np.asarray(expected_std))
    name, ranking = choose_alignment(
        np.asarray(observed_raw), expected_raw, max_mismatch=max_mismatch, min_margin=min_margin,
        allowed=ROTATION_SYMMETRIES,
    )
    if name != IDENTITY:
        inv = inverse_symmetry(name)
        det.bbox_warp = apply_symmetry(det.bbox_warp, inv)
        if det.centers_warp is not None:
            det.centers_warp = apply_symmetry(det.centers_warp, inv)
        if det.centers_img is not None:
            det.centers_img = apply_symmetry(det.centers_img, inv)
    return name, ranking


class ChessVisionTracker:
    def __init__(self, cfg: AppConfig, model=None, clock: Callable[[], float] = time.time):
        self.cfg = cfg
        self.clock = clock

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

        self.det: DetectionResult | None = None

        self.accepted_occ = None
        self.observed_occ = None
        self.last_std_occ = None

        self.init_label_grids: list[np.ndarray] = []
        self.init_conf_grids: list[np.ndarray] = []
        self.initialized = False
        self.init_alignment: str = IDENTITY
        self.init_dist: int = 0

        # frame cache (részleges újraklasszifikáláshoz és a mozgás-jelhez)
        self.prev_warp = None
        self.prev_raw_labels = None
        self.prev_raw_confs = None
        self.prev_raw_types = None
        self.frame_counter = 0
        self._warp_history: deque[tuple[float, np.ndarray]] = deque()
        self._rolling_idx = 0
        self._force_full = False

        # beragadás-kezelés
        self._unsettled_since: float | None = None
        self._last_redetect_t: float = -math.inf
        self._last_wakeup_t: float = -math.inf
        self.redetect_count = 0
        self.last_redetect_alignment: str | None = None

        # a háttér-őrszem által átadott, KÉSZ detektálás
        self._offered_det: tuple[DetectionResult, str] | None = None
        self._offer_lock = threading.Lock()
        self.watchdog_swaps: int = 0
        self.last_watchdog_swap: str | None = None

        # elfogadási politika állapota
        self._pending_promotion: tuple[str, float] | None = None
        self.last_accept_info: dict | None = None

        # Init phase timing (perf_counter-based, ms)
        self._init_start: float | None = None
        self._init_frames: int = 0
        self._init_board_detect_ms: float | None = None
        self._init_total_ms: float | None = None

    # ------------------------------------------------------------------ #
    # modell / profiler                                                    #
    # ------------------------------------------------------------------ #

    def _load_model(self):
        if self.profiler:
            self.profiler.start("model_load")
        from vision.models.occupancy_color_model import OccupancyColorModel
        model = OccupancyColorModel(
            weights_path=self.cfg.weights_path,
            backend=getattr(self.cfg, "inference_backend", "auto"),
            onnx_path=getattr(self.cfg, "onnx_path", None),
            num_threads=getattr(self.cfg, "inference_threads", None),
            allow_int8=getattr(self.cfg, "allow_int8", True),
        )
        print(f"[tracker] {model!r}")
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
            "alignment": self.init_alignment,
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

    # ------------------------------------------------------------------ #
    # eredmény-objektumok                                                  #
    # ------------------------------------------------------------------ #

    def _make_result(self, *, initialized: bool, board_changed: bool, raw_labels, confs_std, san, uci,
                     mode: str, raw_dist: int, obs_mean: float, reason: str = "", motion: float | None = None,
                     accept_info: dict | None = None) -> FrameProcessResult:
        accepted_occ = None if self.accepted_occ is None else self.accepted_occ.copy()
        observed_occ = None if self.observed_occ is None else self.observed_occ.copy()
        return FrameProcessResult(
            initialized=initialized, board_changed=board_changed, accepted_occ=accepted_occ,
            observed_occ=observed_occ, raw_labels=raw_labels, confs_std=confs_std, san=san, uci=uci,
            mode=mode, raw_dist=raw_dist, obs_mean=obs_mean, reason=reason, motion=motion, accept_info=accept_info,
        )

    def _no_move_result(self, raw_labels, confs_std, mode: str, raw_dist: int, obs_mean: float,
                        reason: str = "", motion: float | None = None) -> FrameProcessResult:
        return self._make_result(initialized=True, board_changed=False, raw_labels=raw_labels, confs_std=confs_std,
                                 san=None, uci=None, mode=mode, raw_dist=raw_dist, obs_mean=obs_mean,
                                 reason=reason, motion=motion)

    def _init_result(self, mode: str, raw_labels=None, confs_std=None, obs_mean: float = 0.0,
                     initialized: bool = False) -> FrameProcessResult:
        return self._make_result(initialized=initialized, board_changed=False, raw_labels=raw_labels,
                                 confs_std=confs_std, san=None, uci=None, mode=mode, raw_dist=0, obs_mean=obs_mean)

    # ------------------------------------------------------------------ #
    # klasszifikáció                                                       #
    # ------------------------------------------------------------------ #

    def _warp_frame(self, frame_bgr: np.ndarray):
        with self._profile("warp"):
            return cv2.warpPerspective(frame_bgr, self.det.M, self.cfg.warp_size, flags=cv2.WARP_INVERSE_MAP)

    def _full_classify_warp(self, img_warp: np.ndarray) -> BatchClassificationResult:
        with self._profile("classifier_full"):
            return classify_warp_squares_batch(img_warp, self.det.bbox_warp, self.occ_model, context=self.cfg.context)

    def _full_classify(self, frame_bgr: np.ndarray) -> BatchClassificationResult:
        with self._profile("classifier_full"):
            return classify_frame_batch(frame_bgr, self.det, self.cfg.warp_size, self.occ_model, context=self.cfg.context)

    def _rolling_squares(self) -> list[tuple[int, int]]:
        n = int(self.cfg.rolling_refresh_squares)
        if n <= 0:
            return []
        out = []
        for k in range(n):
            idx = (self._rolling_idx + k) % 64
            out.append((idx // 8, idx % 8))
        self._rolling_idx = (self._rolling_idx + n) % 64
        return out

    def _squares_to_reclassify(self, diffs) -> list[tuple[int, int]]:
        changed = [(r, c) for diff, r, c in diffs[: self.cfg.partial_max_squares] if diff > self.cfg.partial_diff_threshold]
        seen = set(changed)
        for sq in self._rolling_squares():
            if sq not in seen:
                changed.append(sq)
                seen.add(sq)
        return changed

    def _have_cache(self) -> bool:
        return self.prev_warp is not None and self.prev_raw_labels is not None and self.prev_raw_confs is not None

    def _should_full_classify(self) -> bool:
        if not self.cfg.partial_reclassify or not self._have_cache():
            return True
        if self._force_full:
            return True
        return self.frame_counter % self.cfg.full_reclassify_interval == 0

    def _classify(self, img_warp: np.ndarray, diffs) -> BatchClassificationResult:
        if self._should_full_classify():
            self._force_full = False
            return self._full_classify_warp(img_warp)

        squares = self._squares_to_reclassify(diffs)
        labels = self.prev_raw_labels.copy()
        confs = self.prev_raw_confs.copy()
        types = None if self.prev_raw_types is None else self.prev_raw_types.copy()
        if not squares:
            return BatchClassificationResult(labels=labels, confs=confs, type_probs=types)

        with self._profile("classifier_partial"):
            res = classify_squares_batch(img_warp, self.det.bbox_warp, squares, self.occ_model, context=self.cfg.context)
        for i, (r, c) in enumerate(res.positions):
            labels[r, c] = int(res.labels[i])
            confs[r, c] = float(res.confs[i])
            if types is not None and res.type_probs is not None:
                types[r, c] = res.type_probs[i]
        return BatchClassificationResult(labels=labels, confs=confs, type_probs=types)

    # ------------------------------------------------------------------ #
    # mozgás-jel                                                           #
    # ------------------------------------------------------------------ #

    def _older_warp(self, now: float):
        """A legfrissebb olyan cache-elt warp, amely legalább motion_ref_age_s-mal régebbi."""
        cutoff = now - self.cfg.motion_ref_age_s
        chosen = None
        for t, w in self._warp_history:
            if t <= cutoff:
                chosen = w
            else:
                break
        return chosen

    def _motion_and_diffs(self, img_warp: np.ndarray, now: float):
        """(mozgás-jel | None, négyzetenkénti diff-lista az előző frame-hez)."""
        if self.prev_warp is None:
            return None, []
        with self._profile("square_diff"):
            diffs = compute_square_diffs(self.prev_warp, img_warp, self.det.bbox_warp)
            motion = diffs[0][0] if diffs else 0.0
            older = self._older_warp(now)
            if older is not None and older is not self.prev_warp:
                d2 = compute_square_diffs(older, img_warp, self.det.bbox_warp)
                if d2:
                    motion = max(motion, d2[0][0])
        return motion, diffs

    def _update_frame_cache(self, img_warp: np.ndarray, cls: BatchClassificationResult, now: float):
        self.prev_warp = img_warp
        self.prev_raw_labels = cls.labels.copy()
        self.prev_raw_confs = cls.confs.copy()
        self.prev_raw_types = None if cls.type_probs is None else cls.type_probs.copy()
        self.frame_counter += 1
        self._warp_history.append((now, img_warp))
        # csak annyi warp marad a memóriában, amennyi a motion_ref_age_s-os
        # referenciához kell (30 fps-en ~5 db x 8 MB)
        keep_from = now - (self.cfg.motion_ref_age_s * 1.5 + 0.02)
        while len(self._warp_history) > 2 and self._warp_history[0][0] < keep_from:
            self._warp_history.popleft()

    def _drop_frame_cache(self):
        self.prev_warp = None
        self.prev_raw_labels = None
        self.prev_raw_confs = None
        self.prev_raw_types = None
        self._warp_history.clear()

    # ------------------------------------------------------------------ #
    # tábla-detektálás + orientáció-igazítás                               #
    # ------------------------------------------------------------------ #

    def _detect_board(self, gray: np.ndarray):
        with self._profile("board_detect"):
            return detect_board_on_frame(gray, cell=self.cfg.cell, inner_pad_ratio=self.cfg.inner_pad_ratio)

    def _align(self, det: DetectionResult, observed_raw: np.ndarray, expected_std: np.ndarray) -> str:
        if not self.cfg.redetect_align_orientation:
            return IDENTITY
        name, _ = align_detection_to_expected(
            det, observed_raw, expected_std,
            max_mismatch=self.cfg.align_max_mismatch, min_margin=self.cfg.align_min_margin,
        )
        return name

    def detection_snapshot(self) -> tuple[CenterGrid | None, np.ndarray | None]:
        """Olvasható pillanatkép a háttér-őrszemnek: a mezőközepek képpontban
        és az elfogadott állás. Init után a `det`-et már csak cseréljük, sosem
        módosítjuk helyben, ezért ez zár nélkül is biztonságos."""
        det = self.det
        if det is None or det.centers_img is None:
            return None, None
        return det.centers_img, self.accepted_occ

    def align_offered_detection(self, det: DetectionResult, frame_bgr: np.ndarray) -> str:
        """Orientáció-igazítás egy háttérben készült detektáláshoz.

        A detektor rács-orientációja nem garantált (184 mért képből 18-nál jött
        vissza elforgatva), ezért az elfogadott álláshoz igazítjuk. A `det`-et
        helyben rendezi át; a hívó szál a sajátját adja át, a fő szálé érintetlen.
        """
        if self.accepted_occ is None or not self.cfg.redetect_align_orientation:
            return IDENTITY
        cls = classify_frame_batch(frame_bgr, det, self.cfg.warp_size,
                                   self.occ_model, context=self.cfg.context)
        return self._align(det, cls.labels, self.accepted_occ)

    def offer_detection(self, det: DetectionResult, alignment: str) -> None:
        """A háttérszál átad egy KÉSZ, már orientáció-igazított detektálást.

        A tábla-detektálás 340-580 ms, az orientáció-igazítás további ~50 ms —
        ez a fő ciklusban egy fél másodperces kiesést jelentene. Ezért a munka
        a háttérben történik, és ide már csak a kész eredmény érkezik.
        """
        with self._offer_lock:
            self._offered_det = (det, alignment)

    def _consume_offered_detection(self) -> None:
        """Az átadott detektálás beemelése — a fő szálon ez mikroszekundum."""
        with self._offer_lock:
            offered, self._offered_det = self._offered_det, None
        if offered is None:
            return
        det, alignment = offered
        self.det = det
        # A warp elmozdult: a gyorsítótárazott előző képkocka és címkéi már nem
        # ehhez a rácshoz tartoznak — el kell dobni, hogy teljes átosztályozás legyen.
        self._drop_frame_cache()
        self.stabilizer.note_motion(self.clock())
        self.watchdog_swaps += 1
        self.last_watchdog_swap = alignment

    def _should_redetect_board(self, decision: StabilizerDecision, now: float) -> bool:
        """True once the stabilizer has been unable to settle for long enough
        that the frozen homography is worth re-solving."""
        if decision.reason in SETTLED_REASONS:
            self._unsettled_since = None
            return False
        if self._unsettled_since is None:
            self._unsettled_since = now
            return False
        if (now - self._unsettled_since) < self.cfg.redetect_after_stuck_s:
            return False
        return (now - self._last_redetect_t) >= self.cfg.redetect_min_interval_s

    def _redetect_board(self, frame_bgr: np.ndarray, now: float) -> bool:
        """Re-solves the board homography, re-aligns its orientation to the
        accepted position and invalidates the frame cache."""
        self._last_redetect_t = now
        self._unsettled_since = None
        self.redetect_count += 1

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        det = self._detect_board(gray)
        if not det.ok:
            self.last_redetect_alignment = None
            return False

        # A rács orientációja nem garantált: az elfogadott álláshoz igazítjuk.
        alignment = IDENTITY
        if self.accepted_occ is not None and self.cfg.redetect_align_orientation:
            cls = classify_frame_batch(frame_bgr, det, self.cfg.warp_size, self.occ_model, context=self.cfg.context)
            alignment = self._align(det, cls.labels, self.accepted_occ)
        self.last_redetect_alignment = alignment

        self.det = det
        # The warp moved, so the cached previous frame and its labels no longer
        # line up with the new bboxes — drop them to force a full reclassify.
        self._drop_frame_cache()
        self.stabilizer.note_motion(now)
        return True

    # ------------------------------------------------------------------ #
    # init                                                                 #
    # ------------------------------------------------------------------ #

    def _reset_init_buffers(self):
        self.det = None
        self.init_label_grids.clear()
        self.init_conf_grids.clear()

    def _store_init_baseline(self, frame_bgr: np.ndarray, cls, init_labels_std, init_confs_std, now: float):
        self.accepted_occ = init_labels_std.copy()
        self.observed_occ = init_labels_std.copy()
        self.last_std_occ = init_labels_std.copy()
        self.stabilizer.set_reference(init_labels_std.tolist())
        self.stabilizer.update(init_labels_std.tolist(), init_confs_std.tolist(), now_s=now, motion=None)
        self.initialized = True

        img_warp = cv2.warpPerspective(frame_bgr, self.det.M, self.cfg.warp_size, flags=cv2.WARP_INVERSE_MAP)
        self.frame_counter = 0
        self._update_frame_cache(img_warp, cls, now)

    def try_initialize_from_frame(self, frame_bgr: np.ndarray) -> FrameProcessResult:
        now = self.clock()
        if self._init_start is None:
            self._init_start = time.perf_counter()
        self._init_frames += 1

        if self.det is None:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            t_det = time.perf_counter()
            det = self._detect_board(gray)
            if not det.ok:
                return self._init_result("detect-failed")
            if self._init_board_detect_ms is None:
                self._init_board_detect_ms = (time.perf_counter() - t_det) * 1000.0
            self.det = det
            # Orientáció-igazítás az induló álláshoz (csak forgatások — az
            # alapállás bal-jobb tükrözésre szimmetrikus, azt nem döntjük el).
            cls0 = self._full_classify(frame_bgr)
            self.init_alignment = self._align(self.det, cls0.labels, self.expected_start_occ)
            if self.init_alignment != IDENTITY:
                cls0 = self._full_classify(frame_bgr)
            cls = cls0
        else:
            cls = self._full_classify(frame_bgr)

        self.init_label_grids.append(cls.labels)
        self.init_conf_grids.append(cls.confs)

        if len(self.init_label_grids) < self.cfg.init_buffer_frames:
            return self._init_result(f"init-buffer {len(self.init_label_grids)}/{self.cfg.init_buffer_frames}",
                                     raw_labels=cls.labels)

        init_labels, init_confs = weighted_vote_occ(
            [raw_to_standard(g) for g in self.init_label_grids],
            [raw_to_standard(g) for g in self.init_conf_grids],
        )
        init_dist = occ_distance(init_labels, self.expected_start_occ)
        if init_dist > self.cfg.init_max_dist:
            self._reset_init_buffers()
            return self._init_result(f"init-too-far dist={init_dist}", raw_labels=cls.labels)

        self._store_init_baseline(frame_bgr, cls, init_labels, init_confs, now)
        confs_std = raw_to_standard(cls.confs)
        self.init_dist = init_dist
        if init_dist > 0:
            # A megfigyelt tábla eltér a várt kezdőállástól (init_max_dist-en
            # belül). A resolver a Game állásához képest keres lépést, ezért egy
            # rosszul felrakott bábu MINDEN további lépést feloldhatatlanná tesz
            # — ezt látni kell az overlay-en / logban.
            cells = [(int(r), int(c)) for r, c in zip(*np.nonzero(init_labels != self.expected_start_occ))]
            print(f"[tracker] FIGYELEM: az init-állás {init_dist} mezőben eltér a várt kezdőállástól "
                  f"(standard sor,oszlop): {cells} — ellenőrizd a felrakást!")

        if self._init_start is not None:
            self._init_total_ms = (time.perf_counter() - self._init_start) * 1000.0
            if self.profiler:
                self.profiler.record("init_total", self._init_total_ms)
                if self._init_board_detect_ms is not None:
                    self.profiler.record("init_board_detect_ok", self._init_board_detect_ms)

        return self._init_result("init-ok" if init_dist == 0 else f"init-ok dist={init_dist}",
                                 raw_labels=cls.labels, confs_std=confs_std,
                                 obs_mean=float(np.mean(init_confs)), initialized=True)

    # ------------------------------------------------------------------ #
    # feloldás / elfogadás                                                 #
    # ------------------------------------------------------------------ #

    def _stabilize_observation(self, labels_std, confs_std, now: float, motion: float | None) -> StabilizerDecision:
        with self._profile("stabilizer"):
            return self.stabilizer.update(labels_std.tolist(), confs_std.tolist(), now_s=now, motion=motion)

    def _resolve_move(self, stable_occ: np.ndarray, confs_std: np.ndarray, types_std):
        with self._profile("resolve"):
            return self.game.resolve_from_occupancy(
                stable_occ.tolist(),
                confs_std.tolist(),
                max_noise_cells=self.cfg.fuzzy_max_noise_cells,
                max_weighted_cost=self.cfg.fuzzy_max_weighted_cost,
                min_changed_cells=2,
                type_probs=types_std,
                promotion_min_conf=self.cfg.promotion_min_conf,
                use_type_hint_for_moves=self.cfg.use_type_hint_for_moves,
            )

    def _apply_move(self, best_uci: str):
        with self._profile("apply_move"):
            return self.game.apply_uci(best_uci)

    def _request_wakeup_reclassify(self, now: float) -> None:
        """Eseményvezérelt teljes átosztályozás (rate-limitelve)."""
        if (now - self._last_wakeup_t) >= self.cfg.wakeup_full_reclassify_s:
            self._force_full = True
            self._last_wakeup_t = now

    def _promotion_still_pawn(self, types_std, to_row: int, to_col: int) -> bool:
        """True, ha a típus-fej a célmezőn még GYALOGOT lát (a játékos még nem cserélt)."""
        if types_std is None:
            return False
        v = np.asarray(types_std[to_row][to_col], dtype=np.float32)
        return int(v.argmax()) == TYPE_INDEX["pawn"] and float(v.max()) >= self.cfg.promotion_min_conf

    def process_frame(self, frame_bgr: np.ndarray) -> FrameProcessResult:
        self._consume_offered_detection()

        if not self.initialized:
            return self.try_initialize_from_frame(frame_bgr)

        now = self.clock()
        img_warp = self._warp_frame(frame_bgr)
        motion, diffs = self._motion_and_diffs(img_warp, now)
        cls = self._classify(img_warp, diffs)

        raw_labels = cls.labels
        labels_std = raw_to_standard(cls.labels)
        confs_std = raw_to_standard(cls.confs)
        types_std = None if cls.type_probs is None else raw_to_standard(cls.type_probs)

        self.observed_occ = labels_std
        obs_mean = float(np.mean(confs_std))

        raw_dist = disturbance_score(self.last_std_occ, labels_std)
        self.last_std_occ = labels_std

        self._update_frame_cache(img_warp, cls, now)

        decision = self._stabilize_observation(labels_std, confs_std, now, motion)

        # Feloldhatatlan tartós eltérés (1 mezős) -> célzott ébresztés: teljes átosztályozás.
        if decision.mode == "NOISE":
            self._request_wakeup_reclassify(now)

        if self._should_redetect_board(decision, now):
            ok = self._redetect_board(frame_bgr, now)
            mode = "redetect-ok" if ok else "redetect-failed"
            reason = f"align={self.last_redetect_alignment}" if ok else "detect-failed"
            return self._no_move_result(raw_labels, confs_std, mode, raw_dist, obs_mean, reason, motion)

        if decision.emit_occ is None:
            return self._no_move_result(raw_labels, confs_std, f"{decision.mode}:{decision.reason}", raw_dist,
                                        obs_mean, decision.reason, motion)

        stable_occ = np.asarray(decision.emit_occ, dtype=np.int32)
        if occ_distance(self.accepted_occ, stable_occ) == 0:
            return self._no_move_result(raw_labels, confs_std, "stable-same", raw_dist, obs_mean, decision.reason, motion)

        resolve_result = self._resolve_move(stable_occ, confs_std, types_std)
        if resolve_result.move is None:
            # Stabil, de legális lépés nélküli állapot: rossz címke gyanús -> átosztályozás.
            self._request_wakeup_reclassify(now)
            return self._no_move_result(raw_labels, confs_std, "no-legal-fit", raw_dist, obs_mean,
                                        f"no_legal_fit(changed={len(decision.changed_cells)})", motion)

        best_uci = resolve_result.move.to_uci()

        # Előtag-kétértelműség (bástyával kezdett sánc): hosszabb megerősítés.
        if resolve_result.ambiguous_with:
            need = self.stabilizer.params.min_stable_s + self.cfg.prefix_ambiguity_extra_s
            if decision.stable_s < need:
                return self._no_move_result(raw_labels, confs_std, "ambiguous-wait", raw_dist, obs_mean,
                                            f"ambiguous_with={','.join(resolve_result.ambiguous_with)} "
                                            f"({decision.stable_s * 1000:.0f}/{need * 1000:.0f}ms)", motion)

        # Promóció: ha a célmezőn még gyalog áll, várunk a cserére (max promotion_wait_s).
        if resolve_result.move.promotion_piece:
            key = best_uci[:4]
            if self._promotion_still_pawn(types_std, resolve_result.move.to_row, resolve_result.move.to_col):
                if self._pending_promotion is None or self._pending_promotion[0] != key:
                    self._pending_promotion = (key, now)
                if now - self._pending_promotion[1] < self.cfg.promotion_wait_s:
                    return self._no_move_result(raw_labels, confs_std, "promotion-wait", raw_dist, obs_mean,
                                                f"promotion_pawn_still_on_{key[2:]}", motion)
        self._pending_promotion = None

        applied_move = self._apply_move(best_uci)
        if applied_move is None:
            return self._no_move_result(raw_labels, confs_std, "illegal-reject", raw_dist, obs_mean, decision.reason, motion)

        san = self.game.move_history[-1].san if self.game.move_history else None
        self.accepted_occ = np.asarray(resolve_result.expected_occ, dtype=np.int32)
        self.stabilizer.set_reference(self.accepted_occ.tolist())
        self._unsettled_since = None

        self.last_accept_info = {
            "t_accept": now,
            "candidate_since": decision.candidate_since,
            "last_motion_t": decision.last_motion_t,
            "stable_s": decision.stable_s,
            "static_s": decision.static_s,
            "changed_cells": len(decision.changed_cells),
        }

        return self._make_result(
            initialized=True, board_changed=True, raw_labels=raw_labels, confs_std=confs_std,
            san=san, uci=best_uci, mode=resolve_result.mode or "exact", raw_dist=raw_dist, obs_mean=obs_mean,
            reason=decision.reason, motion=motion, accept_info=dict(self.last_accept_info),
        )
