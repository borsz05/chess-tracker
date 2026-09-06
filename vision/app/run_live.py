from __future__ import annotations

import csv
import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib import error, request

import cv2

from vision.app.config import AppConfig
from vision.app.debug_draw import build_overlay_lines, draw_lines, draw_square_class_dots
from vision.pipeline.tracker import ChessVisionTracker

logger = logging.getLogger(__name__)

_CALIBRATION_FILE = Path(__file__).resolve().parents[2] / "robot" / "calibration.json"
_CAMERA_FAIL_LIMIT = 50


def _check_calibration_gate() -> None:
    """Refuse to start if robot/calibration.json is missing.

    The operator must run `python -m robot.calibrate` first, then clear
    the board of hands and the robot arm before starting vision detection.
    Pass --no-robot to skip this check when running without the robot.
    """
    if "--no-robot" in sys.argv:
        return
    if not _CALIBRATION_FILE.exists():
        logger.error(
            "\nERROR: robot/calibration.json not found.\n"
            "Run calibration first:\n"
            "    python -m robot.calibrate\n"
            "Then ensure the board is clear and restart the vision pipeline.\n"
            "To run without a robot (vision-only), pass --no-robot.\n"
        )
        sys.exit(1)
    logger.info("Calibration file found: %s", _CALIBRATION_FILE)
    ans = input(
        "Calibration complete and board clear of hands/robot? [Y/n]: "
    ).strip().lower()
    if ans not in ("", "y"):
        logger.info("Aborting — re-run after clearing the board.")
        sys.exit(0)


@dataclass
class LiveConfig:
    camera_index: int = 4
    # 1080p: egy mező ~109 kamera-pixel a 720p-s ~72,5 helyett (2,25x több
    # információ). A per-frame költség nem nő, mert a warp így is a fix
    # 1632x1632-re megy (mérve: 1,9 ms mindkét felbontáson); csak az egyszeri
    # board_detect lassul 1242 -> 1801 ms.
    camera_width: int = 1920
    camera_height: int = 1080
    camera_fps: int = 30

    # 30 fps capture / 3 = 10 fps processing (~100 ms budget). Measured
    # frame_total is p50 8 ms, p95 66 ms, so this fits with margin; the old
    # value of 6 was tuned before partial reclassify existed and just added
    # latency to every stabilizer gate.
    process_every_nth_captured_frame: int = 3
    preview_max_width: int = 1280

    show_preview: bool = True
    show_status_overlay: bool = True
    show_square_class_debug: bool = True
    print_accepts: bool = True

    auto_reset_on_init_detect_fail_streak: int = 60

    backend_enabled: bool = True
    backend_origin: str = os.getenv("BACKEND_URL", "http://127.0.0.1:8001")
    backend_timeout_s: float = 2.5
    reset_backend_on_start: bool = True
    reset_backend_on_manual_tracker_reset: bool = True

    enable_timing: bool = True
    timing_output_dir: str = "timing_output"
    # Minimum disturbance (changed cells) to mark a move as "started"
    move_disturbance_threshold: int = 2


class BackendSyncClient:
    def __init__(self, origin: str, timeout_s: float):
        self.origin = origin.rstrip("/")
        self.timeout_s = timeout_s

    def _request_json(self, method: str, path: str, payload: dict | None = None) -> dict:
        body = None
        headers = {}

        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = request.Request(
            url=f"{self.origin}{path}",
            data=body,
            headers=headers,
            method=method,
        )

        try:
            with request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read()
        except error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {e.code}: {detail}") from e
        except error.URLError as e:
            raise RuntimeError(f"Backend nem elerheto: {e.reason}") from e

        if not raw:
            return {}

        return json.loads(raw.decode("utf-8"))

    def new_game(self) -> dict:
        return self._request_json("POST", "/api/new-game")

    def push_move(self, uci: str) -> dict:
        return self._request_json("POST", "/api/move", {"uci": uci})

    def is_robot_busy(self) -> bool:
        try:
            resp = self._request_json("GET", "/api/robot/busy")
            return bool(resp.get("busy", False))
        except Exception:
            return False


class LatestFrameCamera:
    def __init__(self, camera_index: int, width: int, height: int, fps: int):
        self.cap = self._open_camera(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("Nem sikerült megnyitni a kamerát.")

        # MJPG KELL a felbontás előtt: 1920x1080-on a kamera csak MJPG-vel ad
        # 30 fps-t (YUYV-ban nincs is 30 fps-es mód), és OpenCV alapból YUYV-ot
        # választhat -> néhány fps-re esne a képfrissítés.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_seq = 0
        self._running = False
        self._thread = None

        self._fps_lock = threading.Lock()
        self._fps_counter = 0
        self._fps_last_t = time.time()
        self._capture_fps = 0.0

    @staticmethod
    def _camera_backends():
        if os.name == "nt":
            return [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]
        return [cv2.CAP_ANY]

    @classmethod
    def _open_camera(cls, camera_index: int):
        for backend in cls._camera_backends():
            cap = cv2.VideoCapture(camera_index, backend)
            if cap.isOpened():
                logger.info("Kamera megnyitva index=%s, backend=%s", camera_index, backend)
                return cap
            cap.release()

        return cv2.VideoCapture(camera_index)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        return self

    def _reader_loop(self):
        consecutive_failures = 0
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= _CAMERA_FAIL_LIMIT:
                    with self._lock:
                        self._latest_frame = None
                    logger.error("Camera disconnected — waiting for reconnect")
                    consecutive_failures = 0
                    time.sleep(1.0)
                else:
                    time.sleep(0.01)
                continue
            consecutive_failures = 0
            with self._lock:
                self._latest_seq += 1
                self._latest_frame = frame

            with self._fps_lock:
                self._fps_counter += 1
                now = time.time()
                dt = now - self._fps_last_t
                if dt >= 1.0:
                    self._capture_fps = self._fps_counter / dt
                    self._fps_counter = 0
                    self._fps_last_t = now

    def get_latest(self):
        with self._lock:
            if self._latest_frame is None:
                return None, 0
            return self._latest_frame.copy(), self._latest_seq

    def get_capture_fps(self) -> float:
        with self._fps_lock:
            return self._capture_fps

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.cap.release()


class LiveProcessor:
    def __init__(self, app_cfg: AppConfig, live_cfg: LiveConfig):
        self.app_cfg = app_cfg
        self.live_cfg = live_cfg

        self._tracker_lock = threading.Lock()
        self.tracker = ChessVisionTracker(app_cfg)

        self.backend = None
        if live_cfg.backend_enabled:
            self.backend = BackendSyncClient(
                origin=live_cfg.backend_origin,
                timeout_s=live_cfg.backend_timeout_s,
            )

        self._state_lock = threading.Lock()
        self._reset_runtime_state()

        self._fps_counter = 0
        self._fps_last_t = time.time()

        self._running = False
        self._thread = None

        # Persistent across tracker resets — all moves from the whole session
        self._move_records: list[dict] = []

    def _reset_runtime_state(self):
        self.last_result = None
        self.last_processed_seq = 0
        self.last_accept_san = None
        self.last_accept_uci = None
        self.last_accept_mode = None
        self.last_accept_time = None
        self.processed_frames = 0
        self.avg_process_ms = 0.0
        self.processing_fps = 0.0
        self.last_error = None
        self.init_detect_fail_streak = 0
        self.backend_status = "disabled" if self.backend is None else "idle"
        self.last_backend_error = None
        # Robot busy tracking
        self._robot_was_busy = False
        self._robot_busy_cache = False
        self._robot_busy_last_check = 0.0
        # Timing state
        self._init_start_wall: float = time.time()
        self._init_done_wall: float | None = None
        self._prev_initialized: bool = False
        self._move_disturbance_t0: float | None = None
        self._move_disturbance_frames: int = 0

    def start(self, camera: LatestFrameCamera):
        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, args=(camera,), daemon=True)
        self._thread.start()
        return self

    def _reset_tracker(self):
        with self._tracker_lock:
            self.tracker = ChessVisionTracker(self.app_cfg)
        with self._state_lock:
            self._reset_runtime_state()

    def _set_backend_status(self, status: str, err: str | None = None):
        with self._state_lock:
            self.backend_status = status
            self.last_backend_error = err

    def sync_backend_new_game(self):
        if self.backend is None:
            return

        try:
            self.backend.new_game()
            self._set_backend_status("new-game synced", None)
        except Exception as e:
            self._set_backend_status("new-game failed", repr(e))

    def _push_move_to_backend(self, uci: str):
        if self.backend is None:
            return

        try:
            self.backend.push_move(uci)
            self._set_backend_status(f"move synced: {uci}", None)
        except Exception as e:
            self._set_backend_status(f"move sync failed: {uci}", repr(e))

    def request_reset(self, sync_backend: bool = False):
        self._reset_tracker()
        if sync_backend:
            self.sync_backend_new_game()

    def _should_process_seq(self, seq: int) -> bool:
        with self._state_lock:
            last_processed_seq = self.last_processed_seq
        return (seq - last_processed_seq) >= self.live_cfg.process_every_nth_captured_frame

    def _process_with_tracker(self, frame):
        with self._tracker_lock:
            tracker = self.tracker
            result = tracker.process_frame(frame)
            move_count = len(tracker.game.move_history)
            initialized = tracker.initialized
        return result, move_count, initialized

    def _record_timing(self, dt_ms: float) -> float:
        self.processed_frames += 1

        if self.processed_frames == 1:
            self.avg_process_ms = dt_ms
        else:
            self.avg_process_ms = 0.9 * self.avg_process_ms + 0.1 * dt_ms

        self._fps_counter += 1
        now = time.time()
        fps_dt = now - self._fps_last_t
        if fps_dt >= 1.0:
            self.processing_fps = self._fps_counter / fps_dt
            self._fps_counter = 0
            self._fps_last_t = now

        return now

    def _update_last_accept(self, result, now: float):
        if not result.board_changed:
            return

        self.last_accept_san = result.san
        self.last_accept_uci = result.uci
        self.last_accept_mode = result.mode
        self.last_accept_time = now

    def _update_init_streak(self, initialized: bool, result) -> bool:
        if (not initialized) and str(result.mode).startswith("detect-failed"):
            self.init_detect_fail_streak += 1
        else:
            self.init_detect_fail_streak = 0

        limit = self.live_cfg.auto_reset_on_init_detect_fail_streak
        return limit > 0 and self.init_detect_fail_streak >= limit

    def _store_result(self, seq: int, result, initialized: bool, dt_ms: float) -> bool:
        with self._state_lock:
            self.last_result = result
            self.last_processed_seq = seq
            now = self._record_timing(dt_ms)
            self._update_last_accept(result, now)
            return self._update_init_streak(initialized, result)

    def _store_processing_error(self, exc: Exception):
        with self._state_lock:
            self.last_error = repr(exc)

    def _handle_successful_process(self, result, seq: int, move_count: int):
        if result.board_changed and self.live_cfg.print_accepts:
            logger.info(
                "Elfogadott lépés: %s | mód: %s | feldolgozott seq: %s | lépésszám: %s",
                result.san, result.mode, seq, move_count,
            )

        if result.board_changed and result.uci:
            self._push_move_to_backend(result.uci)

    def _check_robot_busy(self) -> bool:
        """Visszaadja hogy a robot éppen mozog-e. Fél másodpercenként frissíti a cache-t."""
        now = time.time()
        if now - self._robot_busy_last_check < 0.5:
            return self._robot_busy_cache
        self._robot_busy_last_check = now
        if self.backend is None:
            self._robot_busy_cache = False
        else:
            self._robot_busy_cache = self.backend.is_robot_busy()
        return self._robot_busy_cache

    def _worker_loop(self, camera: LatestFrameCamera):
        while self._running:
            frame, seq = camera.get_latest()
            if frame is None or seq == 0:
                time.sleep(0.005)
                continue

            if not self._should_process_seq(seq):
                time.sleep(0.002)
                continue

            robot_busy = self._check_robot_busy()

            # Ha a robot mozog: skip minden frame-et, ne zavarjuk össze a detektálást
            if robot_busy:
                self._robot_was_busy = True
                with self._state_lock:
                    self.backend_status = "robot-moving"
                time.sleep(0.05)
                continue

            # Robot éppen befejezte a mozgást → tracker reset az új pozícióból
            if self._robot_was_busy and not robot_busy:
                self._robot_was_busy = False
                logger.info("Robot kész — tracker reset az új pozícióból.")
                self._reset_tracker()
                time.sleep(1.0)  # 1 másodperc türelmi idő mielőtt újra detektálunk
                continue

            t0 = time.time()
            try:
                result, move_count, initialized = self._process_with_tracker(frame)
            except Exception as e:
                self._store_processing_error(e)
                time.sleep(0.01)
                continue

            t_done = time.time()
            dt_ms = (t_done - t0) * 1000.0

            # Record total frame processing time in profiler
            with self._tracker_lock:
                profiler = self.tracker.profiler
            if profiler:
                profiler.record("frame_total", dt_ms)

            # Init completion detection
            with self._state_lock:
                prev_init = self._prev_initialized
            if not prev_init and initialized:
                init_wall_ms = (t_done - self._init_start_wall) * 1000.0
                with self._tracker_lock:
                    istats = self.tracker.init_stats
                logger.info(
                    "Inicializálás kész! Fal: %.1f ms | Board detect: %s ms | Frames: %d",
                    init_wall_ms,
                    f"{istats['board_detect_ms']:.1f}" if istats and istats.get("board_detect_ms") else "?",
                    istats["frames"] if istats else 0,
                )
                with self._state_lock:
                    self._init_done_wall = t_done
                    self._prev_initialized = True
            elif initialized:
                with self._state_lock:
                    self._prev_initialized = True

            # Move disturbance / end-to-end latency tracking
            if initialized:
                threshold = self.live_cfg.move_disturbance_threshold
                with self._state_lock:
                    disturbance_t0 = self._move_disturbance_t0
                    disturbance_frames = self._move_disturbance_frames

                if result.board_changed:
                    if disturbance_t0 is not None:
                        latency_ms = (t_done - disturbance_t0) * 1000.0
                    else:
                        latency_ms = None
                    self._move_records.append({
                        "move_num": move_count,
                        "uci": result.uci or "",
                        "latency_ms": f"{latency_ms:.2f}" if latency_ms is not None else "",
                        "frames_to_detect": disturbance_frames,
                        "mode": result.mode or "",
                    })
                    with self._state_lock:
                        self._move_disturbance_t0 = None
                        self._move_disturbance_frames = 0
                elif disturbance_t0 is None and result.raw_dist >= threshold:
                    with self._state_lock:
                        self._move_disturbance_t0 = t0
                        self._move_disturbance_frames = 1
                elif disturbance_t0 is not None:
                    with self._state_lock:
                        self._move_disturbance_frames += 1

            do_auto_reset = self._store_result(seq, result, initialized, dt_ms)

            if do_auto_reset:
                self._reset_tracker()

            self._handle_successful_process(result, seq, move_count)

    def snapshot(self):
        with self._state_lock:
            with self._tracker_lock:
                tracker = self.tracker
                initialized = tracker.initialized
                move_count = len(tracker.game.move_history)
                centers_img = tracker.det.centers_img if tracker.det is not None else None

            return {
                "result": self.last_result,
                "last_processed_seq": self.last_processed_seq,
                "last_accept_san": self.last_accept_san,
                "last_accept_uci": self.last_accept_uci,
                "last_accept_mode": self.last_accept_mode,
                "last_accept_time": self.last_accept_time,
                "processed_frames": self.processed_frames,
                "avg_process_ms": self.avg_process_ms,
                "processing_fps": self.processing_fps,
                "last_error": self.last_error,
                "backend_status": self.backend_status,
                "last_backend_error": self.last_backend_error,
                "initialized": initialized,
                "move_count": move_count,
                "centers_img": centers_img,
            }

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self.live_cfg.enable_timing:
            self._export_timing()

    # ------------------------------------------------------------------ #
    # Timing export                                                        #
    # ------------------------------------------------------------------ #

    def _export_timing(self) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(self.live_cfg.timing_output_dir) / ts
        out_dir.mkdir(parents=True, exist_ok=True)

        with self._tracker_lock:
            profiler = self.tracker.profiler
            istats = self.tracker.init_stats

        if profiler:
            profiler.save_samples_csv(out_dir / "components_samples.csv")
            profiler.save_summary_csv(out_dir / "components_summary.csv")
            profiler.report()

        self._save_moves_csv(out_dir / "moves.csv")
        self._print_move_summary(istats)
        logger.info("Időzítési adatok mentve: %s", out_dir.resolve())

    def _save_moves_csv(self, path: Path) -> None:
        if not self._move_records:
            logger.info("Nem volt detektált lépés — moves.csv nem mentve.")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["move_num", "uci", "latency_ms", "frames_to_detect", "mode"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._move_records)
        logger.info("Lépés latenciák mentve: %s (%d lépés)", path, len(self._move_records))

    def _print_move_summary(self, istats: dict | None) -> None:
        import statistics as _statistics
        w = 60
        logger.info("=" * w)
        logger.info("  LÉPÉSDETEKTÁLÁS ÖSSZEFOGLALÓ")
        logger.info("=" * w)

        with self._state_lock:
            init_done = self._init_done_wall
            init_start = self._init_start_wall
        if init_done is not None:
            logger.info("  Inicializálás (fal-idő): %.1f ms", (init_done - init_start) * 1000.0)
        if istats:
            if istats.get("total_ms") is not None:
                logger.info("  Inicializálás (CPU):     %.1f ms", istats["total_ms"])
            if istats.get("board_detect_ms") is not None:
                logger.info("  Board detect (1. ok):    %.1f ms", istats["board_detect_ms"])
            logger.info("  Init frame-ek száma:     %d", istats.get("frames", 0))

        logger.info("-" * w)

        latencies = []
        for r in self._move_records:
            if r.get("latency_ms"):
                try:
                    latencies.append(float(r["latency_ms"]))
                except ValueError:
                    pass

        if not latencies:
            logger.info("  Nincs mérhető lépés latencia.")
            logger.info("=" * w)
            return

        s = sorted(latencies)
        n = len(s)
        logger.info("  Detektált lépések:       %d", n)
        logger.info("  Átlag latencia:          %.1f ms", _statistics.mean(s))
        logger.info("  Min latencia:            %.1f ms", s[0])
        logger.info("  Max latencia:            %.1f ms", s[-1])
        logger.info("  Medián (p50):            %.1f ms", s[n // 2])
        logger.info("  95. percentilis (p95):   %.1f ms", s[min(int(n * 0.95), n - 1)])
        logger.info("=" * w)


def fit_preview(frame, max_width: int):
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame

    scale = max_width / float(w)
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def main():
    logging.basicConfig(
        stream=sys.stdout, level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    _check_calibration_gate()

    app_cfg = AppConfig()
    live_cfg = LiveConfig()

    camera = LatestFrameCamera(
        camera_index=live_cfg.camera_index,
        width=live_cfg.camera_width,
        height=live_cfg.camera_height,
        fps=live_cfg.camera_fps,
    ).start()

    processor = LiveProcessor(app_cfg, live_cfg).start(camera)
    if live_cfg.reset_backend_on_start:
        processor.sync_backend_new_game()

    window_name = "Chess Vision Live"
    if live_cfg.show_preview:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            frame, seq = camera.get_latest()
            if frame is None:
                time.sleep(0.01)
                continue

            state = processor.snapshot()

            if not live_cfg.show_preview:
                time.sleep(0.01)
                continue

            if live_cfg.show_status_overlay:
                lines = build_overlay_lines(
                    state=state,
                    capture_seq=seq,
                    capture_fps=camera.get_capture_fps(),
                    every_nth=live_cfg.process_every_nth_captured_frame,
                )
                draw_lines(frame, lines)

            if live_cfg.show_square_class_debug:
                result = state["result"]
                raw_labels = None if result is None else result.raw_labels
                draw_square_class_dots(frame, state["centers_img"], raw_labels)

            preview = fit_preview(frame, live_cfg.preview_max_width)
            cv2.imshow(window_name, preview)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("r"):
                logger.info("Tracker reset kérve.")
                processor.request_reset(sync_backend=live_cfg.reset_backend_on_manual_tracker_reset)
    finally:
        processor.stop()
        camera.stop()
        if live_cfg.show_preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
