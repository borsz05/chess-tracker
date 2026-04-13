from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from vision.board.find_chessboard import findChessboard, generateNewBestFit, getBestLines
from vision.board.squares import extract_squares_from_warp


BBoxGrid = List[List[Tuple[int, int, int, int]]]
CenterGrid = List[List[Tuple[float, float]]]


def centers_to_image(centers_warp: CenterGrid, M: np.ndarray) -> CenterGrid:
    flat = np.array(
        [centers_warp[r][c] for r in range(8) for c in range(8)],
        dtype=np.float32,
    ).reshape(-1, 1, 2)

    flat_img = cv2.perspectiveTransform(flat, M).reshape(-1, 2)

    out: CenterGrid = [[(0.0, 0.0) for _ in range(8)] for _ in range(8)]
    k = 0
    for r in range(8):
        for c in range(8):
            out[r][c] = (float(flat_img[k, 0]), float(flat_img[k, 1]))
            k += 1

    return out


@dataclass
class DetectionResult:
    ok: bool
    M: Optional[np.ndarray] = None
    bbox_warp: Optional[BBoxGrid] = None
    centers_warp: Optional[CenterGrid] = None
    centers_img: Optional[CenterGrid] = None


def detect_board_on_frame(
    gray: np.ndarray,
    *,
    cell: int,
    inner_pad_ratio: float,
) -> DetectionResult:
    if gray is None or gray.size == 0:
        return DetectionResult(ok=False)

    if gray.ndim != 2:
        raise ValueError("A detect_board_on_frame szürkeárnyalatos képet vár.")

    M0, ideal_grid, grid_next, grid_good, _ = findChessboard(gray.copy())
    if M0 is None or ideal_grid is None or grid_next is None or grid_good is None:
        return DetectionResult(ok=False)

    M = generateNewBestFit((ideal_grid + 8) * cell, grid_next, grid_good)
    if M is None:
        return DetectionResult(ok=False)

    warp_size = (17 * cell, 17 * cell)
    img_warp_gray = cv2.warpPerspective(
        gray,
        M,
        warp_size,
        flags=cv2.WARP_INVERSE_MAP,
    )

    best_x, best_y = getBestLines(img_warp_gray, cell_size=cell)
    if best_x is None or best_y is None:
        return DetectionResult(ok=False)

    bbox_warp, centers_warp, _ = extract_squares_from_warp(
        img_warp_gray,
        best_x,
        best_y,
        inner_pad_ratio=inner_pad_ratio,
    )

    centers_img = centers_to_image(centers_warp, M)

    return DetectionResult(
        ok=True,
        M=M,
        bbox_warp=bbox_warp,
        centers_warp=centers_warp,
        centers_img=centers_img,
    )
