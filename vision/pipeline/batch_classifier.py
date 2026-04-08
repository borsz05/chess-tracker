from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import torch

from vision.models.occupancy_color_model import OccupancyColorModel


LabelArray = np.ndarray   # shape: (8, 8), dtype: int32
ConfArray = np.ndarray    # shape: (8, 8), dtype: float32
BBoxGrid = List[List[Tuple[int, int, int, int]]]


@dataclass
class BatchClassificationResult:
    labels: LabelArray
    confs: ConfArray


def crop_with_context(
    img: np.ndarray,
    bbox: Tuple[int, int, int, int],
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


def preprocess_roi_for_batch(
    roi: np.ndarray,
    model: OccupancyColorModel,
) -> torch.Tensor | None:
    """
    Egy ROI-ból modell input tensor készítése.
    A normalizáció ugyanazt használja, mint amit a checkpoint tárol.
    """
    if roi is None or roi.size == 0:
        return None

    if roi.ndim == 2:
        roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

    size = model.img_size
    h, w = roi.shape[:2]
    interp = cv2.INTER_CUBIC if (w < size or h < size) else cv2.INTER_AREA
    roi = cv2.resize(roi, (size, size), interpolation=interp)

    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    x = torch.from_numpy(np.ascontiguousarray(roi))
    x = x.permute(2, 0, 1).float().div_(255.0)
    x.sub_(model.norm_mean_cpu).div_(model.norm_std_cpu)
    return x


@torch.no_grad()
def classify_warp_squares_batch(
    img_warp: np.ndarray,
    bbox_warp: BBoxGrid,
    model: OccupancyColorModel,
    *,
    context: float = 0.50,
) -> BatchClassificationResult:
    """
    A warpolt képből egyszerre batch-ben klasszifikálja a 64 mezőt.
    """
    labels = np.zeros((8, 8), dtype=np.int32)
    confs = np.zeros((8, 8), dtype=np.float32)

    batch_tensors: list[torch.Tensor] = []
    positions: list[tuple[int, int]] = []

    for r in range(8):
        row_bboxes = bbox_warp[r]
        for c in range(8):
            roi = crop_with_context(img_warp, row_bboxes[c], context=context)
            x = preprocess_roi_for_batch(roi, model)
            if x is None:
                continue
            batch_tensors.append(x)
            positions.append((r, c))

    if not batch_tensors:
        return BatchClassificationResult(labels=labels, confs=confs)

    batch = torch.stack(batch_tensors, dim=0).to(model.device, non_blocking=True)

    logits = model.model(batch)
    probs = torch.softmax(logits, dim=1)

    pred_idx = torch.argmax(probs, dim=1).detach().cpu().numpy()
    pred_conf = probs.max(dim=1).values.detach().cpu().numpy().astype(np.float32)

    label_ids = model.idx_to_label[pred_idx]

    for (r, c), lab, conf in zip(positions, label_ids, pred_conf):
        labels[r, c] = int(lab)
        confs[r, c] = float(conf)

    return BatchClassificationResult(labels=labels, confs=confs)


def classify_frame_batch(
    frame_bgr: np.ndarray,
    det,
    warp_size: Tuple[int, int],
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

@torch.no_grad()
def classify_selected_squares(
    img_warp,
    bbox_grid,
    squares,
    model,
    *,
    context: float = 0.50,
):
    """
    Csak bizonyos mezőket klasszifikál.
    squares: [(r,c),...]
    """
    batch_tensors = []
    mapping = []

    for r, c in squares:
        roi = crop_with_context(img_warp, bbox_grid[r][c], context=context)
        x = preprocess_roi_for_batch(roi, model)
        if x is None:
            continue
        batch_tensors.append(x)
        mapping.append((r, c))

    if not batch_tensors:
        return {}

    batch = torch.stack(batch_tensors, dim=0).to(model.device, non_blocking=True)
    logits = model.model(batch)
    probs = torch.softmax(logits, dim=1)

    pred_idx = torch.argmax(probs, dim=1).detach().cpu().numpy()
    pred_conf = probs.max(dim=1).values.detach().cpu().numpy()
    label_ids = model.idx_to_label[pred_idx]

    return {
        (r, c): (int(lab), float(conf))
        for (r, c), lab, conf in zip(mapping, label_ids, pred_conf)
    }
