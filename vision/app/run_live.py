from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from urllib import error, request

import cv2

from vision.app.config import AppConfig
from vision.pipeline.tracker import ChessVisionTracker


@dataclass
class LiveConfig:
    camera_index: int = 1
    camera_width: int = 1280
    camera_height: int = 720
    camera_fps: int = 30

    process_every_nth_captured_frame: int = 6
    preview_max_width: int = 1280

    show_preview: bool = True
    show_status_overlay: bool = True
    print_accepts: bool = True

    auto_reset_on_init_detect_fail_streak: int = 60

    backend_enabled: bool = True
    backend_origin: str = "http://127.0.0.1:8001"
    backend_timeout_s: float = 2.5
    reset_backend_on_start: bool = True
    reset_backend_on_manual_tracker_reset: bool = True


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


class LatestFrameCamera:
    def __init__(self, camera_index: int, width: int, height: int, fps: int):
        self.cap = self._open_camera(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("Nem sikerült megnyitni a kamerát.")

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

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
                print(f"Kamera megnyitva index={camera_index}, backend={backend}")
                return cap
            cap.release()

        return cv2.VideoCapture(camera_index)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        return self

    def _reader_loop(self):
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

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

    def _get_last_processed_seq(self) -> int:
        with self._state_lock:
            return self.last_processed_seq

    def _should_process_seq(self, seq: int) -> bool:
        last_processed_seq = self._get_last_processed_seq()
        needed_gap = self.live_cfg.process_every_nth_captured_frame
        return (seq - last_processed_seq) >= needed_gap

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
            print(f"Elfogadott lépés: {result.san} | mód: {result.mode} | feldolgozott seq: {seq} | lépésszám: {move_count}")

        if result.board_changed and result.uci:
            self._push_move_to_backend(result.uci)

    def _worker_loop(self, camera: LatestFrameCamera):
        while self._running:
            frame, seq = camera.get_latest()
            if frame is None or seq == 0:
                time.sleep(0.005)
                continue

            if not self._should_process_seq(seq):
                time.sleep(0.002)
                continue

            t0 = time.time()
            try:
                result, move_count, initialized = self._process_with_tracker(frame)
            except Exception as e:
                self._store_processing_error(e)
                time.sleep(0.01)
                continue

            dt_ms = (time.time() - t0) * 1000.0
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
                camera_transform_name = tracker.camera_transform_name

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
                "camera_transform_name": camera_transform_name,
            }

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)


def fit_preview(frame, max_width: int):
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame

    scale = max_width / float(w)
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def draw_lines(frame, lines, x=12, y=28, line_h=24):
    for i, text in enumerate(lines):
        yy = y + i * line_h
        cv2.putText(frame, text, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 1, cv2.LINE_AA)


def build_overlay_lines(state, capture_seq, capture_fps, every_nth):
    result = state["result"]
    lag_frames = max(0, capture_seq - state["last_processed_seq"])

    status = "init" if not state["initialized"] else "tracking"
    mode = result.mode if result is not None else "boot"
    san = state["last_accept_san"] or "-"
    uci = state["last_accept_uci"] or "-"
    san_mode = state["last_accept_mode"] or "-"

    secs_since_accept = "-"
    if state["last_accept_time"] is not None:
        secs_since_accept = f"{time.time() - state['last_accept_time']:.1f}s"

    err = state["last_error"]
    err_text = err[:70] if err else "-"
    backend_status = state["backend_status"]
    backend_err = state["last_backend_error"]
    backend_err_text = backend_err[:70] if backend_err else "-"

    return [
        f"Status: {status}",
        f"Tracker mode: {mode}",
        f"Last accepted: {san} [{uci}] ({san_mode}) | {secs_since_accept} ago",
        f"Moves accepted: {state['move_count']}",
        f"Camera transform: {state['camera_transform_name']}",
        f"Backend sync: {backend_status}",
        f"Capture FPS: {capture_fps:.1f}",
        f"Process FPS: {state['processing_fps']:.1f}",
        f"Avg process time: {state['avg_process_ms']:.0f} ms",
        f"Process every nth captured frame: {every_nth}",
        f"Capture seq: {capture_seq} | Last processed seq: {state['last_processed_seq']} | Gap: {lag_frames}",
        f"Last error: {err_text}",
        f"Backend error: {backend_err_text}",
        "Keys: q = quit, r = reset tracker",
    ]


def main():
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

            if live_cfg.show_status_overlay:
                state = processor.snapshot()
                lines = build_overlay_lines(
                    state=state,
                    capture_seq=seq,
                    capture_fps=camera.get_capture_fps(),
                    every_nth=live_cfg.process_every_nth_captured_frame,
                )
                draw_lines(frame, lines)

            preview = fit_preview(frame, live_cfg.preview_max_width)

            if not live_cfg.show_preview:
                time.sleep(0.01)
                continue

            cv2.imshow(window_name, preview)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("r"):
                print("Tracker reset kérve.")
                processor.request_reset(sync_backend=live_cfg.reset_backend_on_manual_tracker_reset)
    finally:
        processor.stop()
        camera.stop()
        if live_cfg.show_preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
