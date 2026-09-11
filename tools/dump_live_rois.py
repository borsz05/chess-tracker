"""
Élő kamerakép ROI-k mentése — összehasonlításhoz a tanítóképekkel.

Ez a script UGYANAZT a detektálási és kivágási kódutat hívja, amit a
ChessVisionTracker élesben használ:
    - vision.pipeline.board_detector.detect_board_on_frame  (tábla-detektálás + warp)
    - vision.pipeline.batch_classifier.crop_with_context     (mező kivágása kontextussal)
    - AppConfig (cell / inner_pad_ratio / context / weights_path)

Tehát a mentett képek pontosan azok a ROI-k (a resize/normalizálás ELŐTTI
állapotban), amiket a modell ténylegesen megkap éles üzemben. A mentés
konvenciója (r{sor}c{oszlop}, osztály szerinti almappa, alacsony konfidencia
esetén "unsure") megegyezik a tanítóhalmazéval, így a tanítóképek mappája és a
live_dump/ mappa egymás mellé nyithatók, és mezőnként/osztályonként vizuálisan
összevethetők.

Használat (repo gyökérből):
    python -m tools.dump_live_rois
    python -m tools.dump_live_rois --camera-index 4 --out-dir tools/live_dump

Space -> kép mentése (64 mező kivágva, osztályozva, elmentve)
Q     -> kilépés
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Projekt importok — a repo gyökere kerül a sys.path-ra, függetlenül attól,
# hogy `python tools/dump_live_rois.py`-ként vagy `python -m tools.dump_live_rois`-
# ként hívják.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vision.app.config import AppConfig
from vision.app.run_live import LiveConfig
from vision.pipeline.board_detector import detect_board_on_frame
from vision.pipeline.batch_classifier import crop_with_context
from vision.models.occupancy_color_model import OccupancyColorModel

LABEL_TO_NAME = {0: "empty", 1: "white", 2: "black"}


def parse_args() -> argparse.Namespace:
    live_defaults = LiveConfig()
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--camera-index", type=int, default=live_defaults.camera_index,
        help=f"ugyanaz a kamera-index mint run_live.py-ban (alapértelmezett: {live_defaults.camera_index})",
    )
    p.add_argument("--width", type=int, default=live_defaults.camera_width)
    p.add_argument("--height", type=int, default=live_defaults.camera_height)
    p.add_argument("--fps", type=int, default=live_defaults.camera_fps)
    p.add_argument(
        "--out-dir", type=Path, default=ROOT / "tools" / "live_dump",
        help="ide kerülnek a mentett ROI-k, osztály szerinti almappákban",
    )
    p.add_argument(
        "--conf-threshold", type=float, default=0.70,
        help="ez alatt a konfidencia alatt 'unsure' almappába kerül a kép (alapértelmezett: 0.70)",
    )
    p.add_argument(
        "--weights", type=str, default=None,
        help="checkpoint elérési út; alapértelmezés: AppConfig.weights_path (az élesben betöltött modell)",
    )
    return p.parse_args()


def detect_and_warp_color(frame_bgr: np.ndarray, cfg: AppConfig):
    """Ugyanaz a hívási lánc, mint a ChessVisionTracker._detect_board + _full_classify-ban."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    det = detect_board_on_frame(gray, cell=cfg.cell, inner_pad_ratio=cfg.inner_pad_ratio)
    if not det.ok:
        return None, None

    img_warp_color = cv2.warpPerspective(
        frame_bgr, det.M, cfg.warp_size, flags=cv2.WARP_INVERSE_MAP,
    )
    return img_warp_color, det.bbox_warp


def classify_and_save(
    img_warp_color: np.ndarray,
    bbox_warp,
    model: OccupancyColorModel,
    cfg: AppConfig,
    out_dir: Path,
    conf_threshold: float,
    timestamp: str,
) -> int:
    """64 mező kivágása (crop_with_context, ugyanaz mint élesben), osztályozás, mentés."""
    rois, positions = [], []
    for r in range(8):
        for c in range(8):
            roi = crop_with_context(img_warp_color, bbox_warp[r][c], context=cfg.context)
            if roi is None or roi.size == 0:
                continue
            rois.append(roi)
            positions.append((r, c))
    if not rois:
        return 0

    pred = model.predict_rois(rois)

    saved = 0
    for (r, c), roi, label, conf in zip(positions, rois, pred.labels, pred.confs):
        label_name = LABEL_TO_NAME.get(int(label), "unsure")

        if conf < conf_threshold:
            folder = out_dir / "unsure"
            suffix = f"_{label_name}_{conf:.2f}"
        else:
            folder = out_dir / label_name
            suffix = ""

        folder.mkdir(parents=True, exist_ok=True)
        fname = f"{timestamp}_r{r}c{c}{suffix}.jpg"
        cv2.imwrite(str(folder / fname), roi)
        saved += 1

    return saved


def main() -> None:
    args = parse_args()

    cfg = AppConfig()
    weights_path = args.weights or cfg.weights_path
    print(f"Modell betöltése: {weights_path}")
    model = OccupancyColorModel(weights_path=weights_path)
    print(f"Modell betöltve: {model!r}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Kamera nyitása (index={args.camera_index}, {args.width}x{args.height}@{args.fps})...")
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f"Nem sikerült megnyitni a kamerát (index={args.camera_index}).")
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    print("Kamera megnyitva.")
    print("Space = kép mentése (64 mező, ugyanazzal a kóddal mint élesben) | Q = kilépés")
    print(f"Kimeneti mappa: {args.out_dir.resolve()}")

    shot_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        preview = frame.copy()
        overlay_text = f"Shots: {shot_count} | Space = mentes | Q = kilepes"
        cv2.putText(preview, overlay_text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(preview, overlay_text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow("Live ROI dump", preview)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord(" "):
            img_warp_color, bbox_warp = detect_and_warp_color(frame, cfg)
            if img_warp_color is None:
                print("  ✗ Nem sikerült detektálni a táblát.")
                continue

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            saved = classify_and_save(
                img_warp_color, bbox_warp, model, cfg,
                args.out_dir, args.conf_threshold, timestamp,
            )
            shot_count += 1
            print(f"  ✓ {saved} mező elmentve -> {args.out_dir} | Összesen: {shot_count} fotó")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nKész. Összesen {shot_count} fotó készült.")
    print(f"Most nyisd meg egymás mellett: a régi tanítóképek mappáját (pl. data/black, data/white, data/empty)")
    print(f"és a most mentett {args.out_dir}/black, {args.out_dir}/white, {args.out_dir}/empty mappákat —")
    print("ha eltér a méretarány, a fényviszony, vagy a keretezés, az megmagyarázza a rossz 'black' teljesítményt.")


if __name__ == "__main__":
    main()
