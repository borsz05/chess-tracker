#!/usr/bin/env python3
"""
Kamera-felvétel későbbi OFFLINE visszajátszáshoz (tools/replay_frames.py video).

A képkockákat egy külön szál JPEG-ként írja (minőség 95), a fő szál csak
olvassa a kamerát — így a felvétel tartja a 30 fps-t (a cv2.VideoWriter MJPG
kódolása a fő szálban ~8 fps-re fojtotta a rögzítést). Minden frame-hez
időbélyeg kerül a timestamps.csv-be, a visszajátszó ebből rekonstruálja a
valós időt — így a stabilizer fali-idő alapú ablakai offline is ugyanúgy
viselkednek, mint élesben.

Erre való: VALÓDI játék felvétele (kéz, lebegtetés, bástyával kezdett sánc,
promóció), amin a lépés-elfogadási latencia és a hamis elfogadás
számszerűen, ismételhetően mérhető és a küszöbök újrahangolhatók.

Használat (repo gyökérből):

    python -m tools.record_camera --out recordings/jatszma1 --seconds 120
    python -m tools.record_camera --out recordings/jatszma1 --exposure 250 --wb-temp 4600

Utána:
    python -m tools.replay_frames video recordings/jatszma1 --moves "e2e4 e7e5 ..."

Billentyű az előnézetben: q = leállítás.
"""
from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vision.app.run_live import LiveConfig  # noqa: E402


def main() -> None:
    live = LiveConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seconds", type=float, default=0.0, help="0 = amíg q-t nem nyomsz")
    p.add_argument("--camera-index", type=int, default=live.camera_index)
    p.add_argument("--width", type=int, default=live.camera_width)
    p.add_argument("--height", type=int, default=live.camera_height)
    p.add_argument("--fps", type=int, default=live.camera_fps)
    p.add_argument("--exposure", type=float, default=None, help="manuális exponálás (V4L2 egység); auto KI")
    p.add_argument("--wb-temp", type=float, default=None, help="fix fehéregyensúly (K); auto WB KI")
    p.add_argument("--quality", type=int, default=95)
    p.add_argument("--no-preview", action="store_true")
    args = p.parse_args()

    frames_dir = args.out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise SystemExit(f"nem nyitható a kamera (index={args.camera_index})")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if args.exposure is not None:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        cap.set(cv2.CAP_PROP_EXPOSURE, args.exposure)
    if args.wb_temp is not None:
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        cap.set(cv2.CAP_PROP_WB_TEMPERATURE, args.wb_temp)
    for _ in range(10):
        cap.read()
    print(f"[record] {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
          f"exposure={cap.get(cv2.CAP_PROP_EXPOSURE):.0f} auto_exposure={cap.get(cv2.CAP_PROP_AUTO_EXPOSURE):.0f} "
          f"-> {args.out}")

    q: queue.Queue = queue.Queue(maxsize=300)
    ts_f = open(args.out / "timestamps.csv", "w", encoding="utf-8")
    ts_f.write("idx,file,t\n")
    dropped = 0

    def writer():
        while True:
            item = q.get()
            if item is None:
                break
            idx, t, frame = item
            name = f"{idx:06d}.jpg"
            cv2.imwrite(str(frames_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
            ts_f.write(f"{idx},{name},{t:.6f}\n")

    th = threading.Thread(target=writer, daemon=True)
    th.start()

    t0 = time.time()
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            t = time.time()
            try:
                q.put_nowait((idx, t, frame))
                idx += 1
            except queue.Full:
                dropped += 1
            if not args.no_preview:
                small = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
                cv2.putText(small, f"REC {idx} frames  {t - t0:.1f}s  dropped={dropped}", (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("record_camera (q = stop)", small)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
            if args.seconds and (t - t0) >= args.seconds:
                break
    finally:
        q.put(None)
        th.join()
        ts_f.close()
        cap.release()
        cv2.destroyAllWindows()
    el = time.time() - t0
    print(f"[record] {idx} frame {el:.1f} s alatt ({idx / max(el, 1e-6):.1f} fps), eldobva: {dropped}")


if __name__ == "__main__":
    main()
