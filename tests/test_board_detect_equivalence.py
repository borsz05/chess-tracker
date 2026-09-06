"""
A vektorizált tábladetektor (vision/board/find_chessboard.py: nonmax_sup,
findGoodPoints) ugyanazt adja, mint a régi Python-ciklusos megvalósítás —
a 10. szakasz 4. pontja (board_detect 1450-2200 ms -> ~250 ms) NEM változtathat
a detektálás eredményén. A referencia-implementációk itt, a tesztben élnek.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from vision.board import find_chessboard as fc

ROOT = Path(__file__).resolve().parents[1]
DET_CACHE = ROOT / "data_fen" / ".det_cache.pkl"
FRAMES_DIR = ROOT / "data_fen" / "frames"


# ---- referencia (a régi kód szó szerint) ------------------------------------

def _ref_nonmax_sup(img, win=10):
    h, w = img.shape
    img_sup = np.zeros_like(img, dtype=np.float64)
    for i, j in np.argwhere(img):
        ta = max(0, i - win)
        tb = min(h, i + win + 1)
        tc = max(0, j - win)
        td = min(w, j + win + 1)
        cell = img[ta:tb, tc:td]
        if img[i, j] == cell.max():
            img_sup[i, j] = img[i, j]
    return img_sup


def _ref_get_min_saddle_dist(saddle_pts, pt):
    best_dist = None
    best_pt = pt
    for saddle_pt in saddle_pts:
        saddle_pt = saddle_pt[::-1]
        dist = np.sum((saddle_pt - pt) ** 2)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_pt = saddle_pt
    return best_pt, np.sqrt(best_dist)


def _ref_find_good_points(grid, spts, max_px_dist=5):
    new_grid = grid.copy()
    chosen_spts = set()
    N = len(new_grid)
    grid_good = np.zeros(N, dtype=bool)

    def hash_pt(pt):
        return f"{int(pt[0])}_{int(pt[1])}"

    for pt_i in range(N):
        pt2, d = _ref_get_min_saddle_dist(spts, new_grid[pt_i, :2])
        if hash_pt(pt2) in chosen_spts:
            d = max_px_dist
        else:
            chosen_spts.add(hash_pt(pt2))
        if d < max_px_dist:
            new_grid[pt_i, :2] = pt2
            grid_good[pt_i] = True
    return new_grid, grid_good


# ---- szintetikus tesztek -----------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_nonmax_sup_matches_reference(seed):
    rng = np.random.default_rng(seed)
    img = np.zeros((120, 160), dtype=np.float64)
    ys, xs = rng.integers(0, 120, 400), rng.integers(0, 160, 400)
    img[ys, xs] = rng.integers(1, 1_000_000, 400).astype(np.float64)
    # szándékos holtversenyek (azonos érték egy ablakon belül)
    img[10, 10] = img[12, 14] = 777.0
    img[0, 0] = 5.0   # sarok: levágott ablak
    a = fc.nonmax_sup(img, win=10)
    b = _ref_nonmax_sup(img, win=10)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_find_good_points_matches_reference(seed):
    rng = np.random.default_rng(seed)
    spts = rng.integers(0, 1000, (300, 2))                       # [row, col], mint np.argwhere
    n = rng.choice([16, 64, 256])
    grid = rng.uniform(0, 1000, (n, 2)).astype(np.float64)
    # néhány rácspont pontosan egy nyeregpontra / két rácspont ugyanarra a nyeregpontra
    grid[0] = spts[5][::-1] + 0.4
    grid[1] = spts[5][::-1] + 0.6
    grid[2] = spts[7][::-1]
    g1, good1 = fc.findGoodPoints(grid, spts, max_px_dist=5)
    g2, good2 = _ref_find_good_points(grid, spts, max_px_dist=5)
    np.testing.assert_array_equal(good1, good2)
    np.testing.assert_allclose(g1, g2)


def test_find_good_points_empty_inputs():
    grid = np.zeros((4, 2))
    g, good = fc.findGoodPoints(grid, np.zeros((0, 2), dtype=np.int64))
    assert not good.any() and np.array_equal(g, grid)


# ---- valós képek: a teljes detektálás eredménye == a cache-elt régi eredmény --

@pytest.mark.skipif(not (DET_CACHE.exists() and FRAMES_DIR.exists()), reason="nincs data_fen detektálás-cache")
def test_full_detection_matches_cached_reference_on_real_frames():
    """A data_fen/.det_cache.pkl-t a tools/measure_pipeline_noise.py írta a RÉGI
    detektorral; az új detektornak ugyanazt a homográfiát és bbox-rácsot kell adnia."""
    import cv2
    from vision.app.config import AppConfig
    from vision.pipeline.board_detector import detect_board_on_frame

    cfg = AppConfig()
    with open(DET_CACHE, "rb") as f:
        cache = pickle.load(f)
    bases = sorted(cache)[:: max(1, len(cache) // 6)][:6]      # 6 kép, egyenletesen
    assert bases
    for base in bases:
        frame = cv2.imread(str(FRAMES_DIR / f"{base}.jpg"))
        if frame is None:
            continue
        ref = cache[base]
        det = detect_board_on_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cell=cfg.cell, inner_pad_ratio=cfg.inner_pad_ratio)
        assert det.ok == ref.ok, base
        if ref.ok:
            np.testing.assert_allclose(det.M, ref.M, rtol=1e-6, atol=1e-6, err_msg=base)
            assert det.bbox_warp == ref.bbox_warp, base
