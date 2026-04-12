"""
Tanítóadat-gyűjtő script.
Helyezd a _SAKKPROJEKT_FINAL/ gyökérmappájába és futtasd onnan.

Space -> képet készít, detektálja a táblát, kivágja a 64 mezőt,
         osztályozza őket és elmenti a data/ mappába.
Q     -> kilép.

Mappastruktúra:
    data/
        empty/
        white/
        black/
        unsure/
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Projekt importok – a script a gyökérmappából fusson
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from vision.app.config import AppConfig
from vision.board.find_chessboard import findChessboard, generateNewBestFit, getBestLines
from vision.board.squares import extract_squares_from_warp
from vision.pipeline.batch_classifier import crop_with_context, preprocess_roi_for_batch
from vision.models.occupancy_color_model import OccupancyColorModel

import torch

# ---------------------------------------------------------------------------
# Konfig
# ---------------------------------------------------------------------------
CONF_THRESHOLD = 0.70   # ez alatt -> unsure mappa
CONTEXT        = 0.50   # ugyanaz mint AppConfig.context
INNER_PAD      = 0.06   # ugyanaz mint AppConfig.inner_pad_ratio
CELL           = 96     # ugyanaz mint AppConfig.cell
WARP_SIZE      = (17 * CELL, 17 * CELL)

LABEL_TO_NAME = {0: "empty", 1: "white", 2: "black"}

DATA_DIR = ROOT / "data"
for name in ("empty", "white", "black", "unsure"):
    (DATA_DIR / name).mkdir(parents=True, exist_ok=True)

WEIGHTS_PATH = ROOT / "vision" / "models" / "weights" / "resnet18_best_szines_topdown_kepeken.pt"

# ---------------------------------------------------------------------------
# Segédfüggvények
# ---------------------------------------------------------------------------

def detect_and_warp(gray: np.ndarray, frame_bgr: np.ndarray):
    """
    Tábladetektálás + warp, ugyanúgy mint a board_detector.py-ban.
    Visszaad: (img_warp_color, bbox_warp) vagy (None, None) ha sikertelen.
    """
    M0, ideal_grid, grid_next, grid_good, _ = findChessboard(gray.copy())
    if M0 is None:
        return None, None

    M = generateNewBestFit((ideal_grid + 8) * CELL, grid_next, grid_good)
    if M is None:
        return None, None

    img_warp_gray = cv2.warpPerspective(
        gray, M, WARP_SIZE, flags=cv2.WARP_INVERSE_MAP
    )

    best_x, best_y = getBestLines(img_warp_gray, cell_size=CELL)
    if best_x is None or best_y is None:
        return None, None

    _, bbox_warp, _, _ = extract_squares_from_warp(
        img_warp_gray, best_x, best_y, inner_pad_ratio=INNER_PAD
    )

    # színes warp a kivágáshoz (ugyanaz mint a végleges programban)
    img_warp_color = cv2.warpPerspective(
        frame_bgr, M, WARP_SIZE, flags=cv2.WARP_INVERSE_MAP
    )

    return img_warp_color, bbox_warp


@torch.no_grad()
def classify_and_save(
    img_warp_color: np.ndarray,
    bbox_warp,
    model: OccupancyColorModel,
    timestamp: str,
):
    """
    Kivágja a 64 mezőt, osztályozza őket, elmenti a megfelelő mappába.
    Visszaadja a mentett fájlok számát.
    """
    saved = 0

    batch_tensors = []
    rois = []
    positions = []

    for r in range(8):
        for c in range(8):
            roi = crop_with_context(img_warp_color, bbox_warp[r][c], context=CONTEXT)
            x = preprocess_roi_for_batch(roi, model)
            if x is None:
                continue
            batch_tensors.append(x)
            rois.append(roi)
            positions.append((r, c))

    if not batch_tensors:
        return 0

    batch = torch.stack(batch_tensors, dim=0).to(model.device, non_blocking=True)
    logits = model.model(batch)
    probs = torch.softmax(logits, dim=1)

    pred_idx  = torch.argmax(probs, dim=1).cpu().numpy()
    pred_conf = probs.max(dim=1).values.cpu().numpy()
    label_ids = model.idx_to_label[pred_idx]

    for (r, c), roi, label, conf in zip(positions, rois, label_ids, pred_conf):
        label_name = LABEL_TO_NAME.get(int(label), "unsure")

        if conf < CONF_THRESHOLD:
            folder = DATA_DIR / "unsure"
            suffix = f"_{label_name}_{conf:.2f}"
        else:
            folder = DATA_DIR / label_name
            suffix = ""

        fname = f"{timestamp}_r{r}c{c}{suffix}.jpg"
        out_path = folder / fname
        cv2.imwrite(str(out_path), roi)
        saved += 1

    return saved


# ---------------------------------------------------------------------------
# Fő loop
# ---------------------------------------------------------------------------

def main():
    print("Modell betöltése...")
    model = OccupancyColorModel(weights_path=str(WEIGHTS_PATH))
    print(f"Modell betöltve. Device: {model.device}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Nem sikerült megnyitni a kamerát.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("Kamera megnyitva.")
    print("Space = képet készít és elmenti | Q = kilép")

    shot_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        preview = frame.copy()
        cv2.putText(
            preview,
            f"Shots: {shot_count} | Space = fotó | Q = kilép",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA,
        )
        cv2.putText(
            preview,
            f"Shots: {shot_count} | Space = fotó | Q = kilép",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA,
        )
        cv2.imshow("Tanítóadat-gyűjtő", preview)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord(" "):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            print("Tábla detektálása...")

            img_warp_color, bbox_warp = detect_and_warp(gray, frame)

            if img_warp_color is None:
                print("  ✗ Nem sikerült detektálni a táblát.")
                continue

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            saved = classify_and_save(img_warp_color, bbox_warp, model, timestamp)
            shot_count += 1
            print(f"  ✓ {saved} mező elmentve -> data/ | Összesen: {shot_count} fotó")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Kész. Összesen {shot_count} fotó készült.")


if __name__ == "__main__":
    main()