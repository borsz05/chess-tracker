from __future__ import annotations

import time

import cv2


CLASS_COLORS_BGR = {
    0: (120, 120, 120),   # empty
    1: (80, 220, 80),     # white
    2: (80, 80, 220),     # black
}


def draw_lines(frame, lines, x=12, y=28, line_h=24):
    for i, text in enumerate(lines):
        yy = y + i * line_h
        cv2.putText(frame, text, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 1, cv2.LINE_AA)


def build_overlay_lines(state, capture_seq, capture_fps, every_nth, watchdog=None):
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

    motion = getattr(result, "motion", None) if result is not None else None
    reason = getattr(result, "reason", "") if result is not None else ""

    # A háttér-őrszem visszajelzése: enélkül nem lehetne tudni, hogy figyel-e,
    # és hogy egy újradetektálás megtörtént-e.
    if watchdog is None:
        watchdog_text = "kikapcsolva"
    else:
        shift = watchdog.get("last_shift_px")
        watchdog_text = (f"{watchdog.get('checks', 0)} ellenőrzés, "
                         f"{watchdog.get('swaps', 0)} újradetektálás | "
                         f"{watchdog.get('status', '-')}")
        if shift is not None:
            watchdog_text += f" | eltérés {shift:.1f} px"

    return [
        f"Status: {status}",
        f"Tracker mode: {mode}",
        f"Stabilizer: {reason or '-'} | motion: {motion:.1f}" if motion is not None else f"Stabilizer: {reason or '-'}",
        f"Last accepted: {san} [{uci}] ({san_mode}) | {secs_since_accept} ago",
        f"Moves accepted: {state['move_count']}",
        f"Backend sync: {backend_status}",
        f"Capture FPS: {capture_fps:.1f}",
        f"Process FPS: {state['processing_fps']:.1f}",
        f"Avg process time: {state['avg_process_ms']:.0f} ms",
        f"Process every nth captured frame: {every_nth}",
        f"Capture seq: {capture_seq} | Last processed seq: {state['last_processed_seq']} | Gap: {lag_frames}",
        f"Last error: {err_text}",
        f"Backend error: {backend_err_text}",
        f"Tábla-őrszem: {watchdog_text}",
        "Keys: q = quit, r = reset tracker, d = redetect board now",
    ]


def draw_square_class_dots(frame, centers_img, raw_labels, radius: int = 8) -> None:
    if centers_img is None or raw_labels is None:
        return

    for row in range(8):
        for col in range(8):
            cx, cy = centers_img[row][col]
            label = int(raw_labels[row][col])
            color = CLASS_COLORS_BGR.get(label, (0, 255, 255))

            center = (int(round(cx)), int(round(cy)))
            cv2.circle(frame, center, radius + 2, (0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(frame, center, radius, color, thickness=-1, lineType=cv2.LINE_AA)
