from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from vision.models.occupancy_color_model import OccupancyColorModel, Prediction


LabelArray = np.ndarray   # shape: (8, 8), dtype: int32
ConfArray = np.ndarray    # shape: (8, 8), dtype: float32
BBoxGrid = list[list[tuple[int, int, int, int]]]


@dataclass
class BatchClassificationResult:
    labels: LabelArray
    confs: ConfArray
    # Bábutípus-valószínűségek (8, 8, n_type) vagy None legacy modellnél.
    # EGYELŐRE csak promóciónál használható — a fő lépésdetektálás a labels/confs-on megy.
    type_probs: np.ndarray | None = None


def crop_with_context(
    img: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    context: float = 0.50,
) -> np.ndarray:
    """
    A bbox körül kontextust hagyunk.
    context=0.50 azt jelenti, hogy kb. 1.5x-ös teljes méretű crop készül.
    """
    h, w = img.shape[:2]
    x0, y0, x1, y1 = bbox

    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)

    pad_x = int(round(0.5 * context * bw))
    pad_y = int(round(0.5 * context * bh))

    nx0 = max(0, x0 - pad_x)
    ny0 = max(0, y0 - pad_y)
    nx1 = min(w, x1 + pad_x)
    ny1 = min(h, y1 + pad_y)

    if nx1 <= nx0 or ny1 <= ny0:
        return img[y0:y1, x0:x1].copy()

    return img[ny0:ny1, nx0:nx1].copy()


def _crop_many(img_warp: np.ndarray, bbox_grid: BBoxGrid, squares, context: float):
    rois, positions = [], []
    for r, c in squares:
        roi = crop_with_context(img_warp, bbox_grid[r][c], context=context)
        if roi is None or roi.size == 0:
            continue
        rois.append(roi)
        positions.append((r, c))
    return rois, positions


def classify_warp_squares_batch(
    img_warp: np.ndarray,
    bbox_warp: BBoxGrid,
    model: OccupancyColorModel,
    *,
    context: float = 0.50,
) -> BatchClassificationResult:
    """
    A warpolt képből egyszerre batch-ben klasszifikálja a 64 mezőt
    (vektorizált preprocess + egy forward, ONNX vagy torch backend).
    """
    labels = np.zeros((8, 8), dtype=np.int32)
    confs = np.zeros((8, 8), dtype=np.float32)

    rois, positions = _crop_many(img_warp, bbox_warp, [(r, c) for r in range(8) for c in range(8)], context)
    if not rois:
        return BatchClassificationResult(labels=labels, confs=confs)

    pred: Prediction = model.predict_rois(rois)
    type_grid = None
    if pred.type_probs is not None:
        type_grid = np.zeros((8, 8, pred.type_probs.shape[1]), dtype=np.float32)

    for i, (r, c) in enumerate(positions):
        labels[r, c] = int(pred.labels[i])
        confs[r, c] = float(pred.confs[i])
        if type_grid is not None:
            type_grid[r, c] = pred.type_probs[i]

    return BatchClassificationResult(labels=labels, confs=confs, type_probs=type_grid)


def classify_frame_batch(
    frame_bgr: np.ndarray,
    det,
    warp_size: tuple[int, int],
    model: OccupancyColorModel,
    *,
    context: float = 0.50,
) -> BatchClassificationResult:
    """
    Egy teljes frame-ből:
    - warp
    - mezők kivágása
    - batch inferencia
    """
    img_warp_color = cv2.warpPerspective(
        frame_bgr,
        det.M,
        warp_size,
        flags=cv2.WARP_INVERSE_MAP,
    )

    return classify_warp_squares_batch(
        img_warp_color,
        det.bbox_warp,
        model,
        context=context,
    )


@dataclass
class SquareBatchResult:
    """Kiválasztott mezők klasszifikációja (részleges út)."""
    positions: list[tuple[int, int]]
    labels: np.ndarray                 # (N,) int32, pipeline-kódolás
    confs: np.ndarray                  # (N,) float32
    type_probs: np.ndarray | None      # (N, n_type) vagy None legacy modellnél


def classify_squares_batch(
    img_warp: np.ndarray,
    bbox_grid: BBoxGrid,
    squares: list[tuple[int, int]],
    model: OccupancyColorModel,
    *,
    context: float = 0.50,
) -> SquareBatchResult:
    """
    Csak a megadott mezőket klasszifikálja (egy batch), a típus-fej
    kimenetével együtt — a tracker részleges/gördülő újraklasszifikálásához.
    """
    rois, positions = _crop_many(img_warp, bbox_grid, squares, context)
    if not rois:
        return SquareBatchResult([], np.zeros((0,), np.int32), np.zeros((0,), np.float32), None)
    pred = model.predict_rois(rois)
    return SquareBatchResult(positions, pred.labels.astype(np.int32), pred.confs.astype(np.float32), pred.type_probs)
