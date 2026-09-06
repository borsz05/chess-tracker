# Based on https://github.com/Elucidation/ChessboardDetect
# Copyright (c) 2016 Sam — MIT License
# Adapted for Python 3 and trimmed to relevant detection pipeline.

import cv2
import numpy as np

# =========================
# Saddle
# =========================

def getSaddle(gray_img):
    img = gray_img.astype(np.float64)
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    gxx = cv2.Sobel(gx, cv2.CV_64F, 1, 0)
    gyy = cv2.Sobel(gy, cv2.CV_64F, 0, 1)
    gxy = cv2.Sobel(gx, cv2.CV_64F, 0, 1)
    return gxx * gyy - gxy ** 2

def nonmax_sup(img, win=10):
    """Non-maximum suppression: megtartja azokat a nem-nulla pixeleket, amelyek
    a (2*win+1)^2-es ablakukban maximálisak.

    Vektorizált: a max-szűrő egy dilatáció (cv2.dilate) — pontosan ugyanaz,
    mint a régi pixelenkénti Python-ciklus (a szélen az ablak levágva; a
    dilatáció alapértelmezett szegélye a max-ot nem befolyásolja). A régi
    ciklus ~20 ms volt 1080p-n, ez <1 ms. tests/test_board_detect_equivalence.py
    bizonyítja az azonosságot."""
    img = np.asarray(img, dtype=np.float64)
    kernel = np.ones((2 * win + 1, 2 * win + 1), dtype=np.uint8)
    local_max = cv2.dilate(img, kernel)
    img_sup = np.zeros_like(img, dtype=np.float64)
    keep = (img > 0) & (img == local_max)
    img_sup[keep] = img[keep]
    return img_sup

def pruneSaddle(s):
    thresh = 128
    while np.sum(s > 0) > 10000:
        thresh *= 2
        s[s < thresh] = 0

def getMinSaddleDist(saddle_pts, pt):
    best_dist = None
    best_pt = pt

    for saddle_pt in saddle_pts:
        saddle_pt = saddle_pt[::-1]
        dist = np.sum((saddle_pt - pt) ** 2)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_pt = saddle_pt

    return best_pt, np.sqrt(best_dist)

# =========================
# Contours
# =========================

def simplifyContours(contours):
    for i in range(len(contours)):
        contours[i] = cv2.approxPolyDP(
            contours[i],
            0.04 * cv2.arcLength(contours[i], True),
            True
        )

def getAngle(a, b, c):
    k = (a * a + b * b - c * c) / (2 * a * b)
    k = np.clip(k, -1.0, 1.0)
    return np.degrees(np.arccos(k))

def is_square(cnt, eps=3.0):
    center = cnt.sum(axis=0) / 4.0

    dd = [
        np.linalg.norm(cnt[i] - cnt[(i + 1) % 4])
        for i in range(4)
    ]

    xa = np.linalg.norm(cnt[0] - cnt[2])
    xb = np.linalg.norm(cnt[1] - cnt[3])
    xratio = min(xa, xb) / max(xa, xb)

    angles = np.array([
        getAngle(dd[i - 1], dd[i], xb if i % 2 == 0 else xa)
        for i in range(4)
    ])

    good_angles = np.all((angles > 40) & (angles < 140))
    return good_angles and xratio > 0.5

def pruneContours(contours, hierarchy, saddle):
    new_contours = []
    new_hierarchy = []

    for cnt, h in zip(contours, hierarchy):
        if h[2] != -1:
            continue
        if len(cnt) != 4:
            continue
        if cv2.contourArea(cnt) < 64:
            continue
        if not is_square(cnt.squeeze()):
            continue

        cnt = updateCorners(cnt, saddle)
        if len(cnt) == 4:
            new_contours.append(cnt)
            new_hierarchy.append(h)

    if not new_contours:
        return np.array([]), np.array([])

    areas = np.array([cv2.contourArea(c) for c in new_contours])
    med = np.median(areas)
    mask = (areas >= med * 0.25) & (areas <= med * 2.0)

    return np.array(new_contours, dtype=object)[mask], np.array(new_hierarchy)[mask]

def getContours(img, edges):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, kernel)

    contours, hierarchy = cv2.findContours(
        grad, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    contours = list(contours)
    simplifyContours(contours)

    return np.array(contours, dtype=object), hierarchy[0]

# =========================
# Corners
# =========================

def updateCorners(contour, saddle):
    ws = 4
    new_contour = contour.copy()

    for i in range(4):
        x, y = contour[i, 0]
        window = saddle[
            max(0, y - ws): y + ws + 1,
            max(0, x - ws): x + ws + 1
        ]

        if window.size == 0:
            return []

        dy, dx = np.unravel_index(window.argmax(), window.shape)
        if window[dy, dx] > 0:
            new_contour[i, 0] = [x + dx - ws, y + dy - ws]
        else:
            return []

    return new_contour

# =========================
# Grid / Homography
# =========================

def getIdentityGrid(N):
    a = np.arange(N)
    aa, bb = np.meshgrid(a, a)
    return np.vstack([aa.flatten(), bb.flatten()]).T

def findGoodPoints(grid, spts, max_px_dist=5):
    """Minden rácsponthoz a legközelebbi nyeregpont; ha az max_px_dist-en belül
    van és még nem foglalta le korábbi rácspont, a rácspont odaugrik.

    Vektorizált: a teljes (N_grid x N_saddle) távolságmátrix egy numpy
    műveletben, az argmin az első minimumot adja (mint a régi `<` ciklus).
    A "már lefoglalt nyeregpont" szabály sorrendfüggő, ezért az a rész
    marad egy olcsó, N_grid hosszú ciklus. Régen 17-290 ms/hívás volt
    (rácsméret szerint), ez ~1 ms — a board_detect ebből ment 2,2 s-ról
    ~0,25 s-ra. tests/test_board_detect_equivalence.py a régi
    implementációval veti össze."""
    new_grid = grid.copy()
    N = len(new_grid)
    grid_good = np.zeros(N, dtype=bool)
    if N == 0 or len(spts) == 0:
        return new_grid, grid_good

    # spts: (M, 2) [row, col] -> (x, y), mint a régi getMinSaddleDist [::-1]
    spts_xy = np.asarray(spts)[:, ::-1].astype(np.float64)
    pts = np.asarray(new_grid[:, :2], dtype=np.float64)
    d2 = (pts[:, 0:1] - spts_xy[None, :, 0]) ** 2 + (pts[:, 1:2] - spts_xy[None, :, 1]) ** 2
    nearest = d2.argmin(axis=1)
    dist = np.sqrt(d2[np.arange(N), nearest])

    chosen = set()
    for pt_i in range(N):
        k = int(nearest[pt_i])
        if k in chosen:
            d = max_px_dist
        else:
            chosen.add(k)
            d = dist[pt_i]
        if d < max_px_dist:
            new_grid[pt_i, :2] = spts_xy[k]
            grid_good[pt_i] = True

    return new_grid, grid_good

def getInitChessGrid(quad):
    quadA = np.array([[0,1],[1,1],[1,0],[0,0]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quadA, quad.astype(np.float32))
    return makeChessGrid(M, 1)

def makeChessGrid(M, N):
    ideal_grid = getIdentityGrid(2 + 2 * N) - N
    ideal_grid_pad = np.pad(
        ideal_grid,
        ((0, 0), (0, 1)),
        constant_values=1
    )

    grid = (M @ ideal_grid_pad.T).T
    grid[:, :2] /= grid[:, 2:3]
    grid = grid[:, :2]

    return grid, ideal_grid, M

def generateNewBestFit(grid_ideal, grid, grid_good):
    a = grid_ideal[grid_good].astype(np.float32)
    b = grid[grid_good].astype(np.float32)
    M, _ = cv2.findHomography(a, b, cv2.RANSAC)
    return M

def getGrads(img):
    img = cv2.blur(img, (5, 5))
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    grad_mag = gx * gx + gy * gy
    return grad_mag, gx, gy

def getBestLines(img_warped, cell_size=32):
    grad_mag, gx, gy = getGrads(img_warped)

    gx_pos = np.clip(gx, 0, None)
    gx_neg = np.clip(-gx, 0, None)
    score_x = np.sum(gx_pos, axis=0) * np.sum(gx_neg, axis=0)

    gy_pos = np.clip(gy, 0, None)
    gy_neg = np.clip(-gy, 0, None)
    score_y = np.sum(gy_pos, axis=1) * np.sum(gy_neg, axis=1)

    a = np.array(
        [(offset + np.arange(7) + 1) * cell_size for offset in np.arange(1, 9)],
        dtype=np.int32,
    )
    scores_x = np.array([np.sum(score_x[pts]) for pts in a])
    scores_y = np.array([np.sum(score_y[pts]) for pts in a])

    return a[scores_x.argmax()], a[scores_y.argmax()]

def findChessboard(img, min_pts_needed=15, max_pts_needed=25):
    blur_img = cv2.blur(img, (3, 3))
    saddle = -getSaddle(blur_img)
    saddle[saddle < 0] = 0
    pruneSaddle(saddle)

    s2 = nonmax_sup(saddle)
    s2[s2 < 100000] = 0
    spts = np.argwhere(s2)

    edges = cv2.Canny(img, 20, 250)
    contours_all, hierarchy = getContours(img, edges)
    contours, hierarchy = pruneContours(contours_all, hierarchy, saddle)

    curr_num_good = 0
    curr_grid_next = curr_grid_good = curr_M = None
    num_good = 0

    for cnt in contours:
        cnt = cnt.squeeze()
        grid_curr, ideal_grid, M = getInitChessGrid(cnt)

        for grid_i in range(7):
            grid_curr, ideal_grid, _ = makeChessGrid(M, grid_i + 1)
            grid_next, grid_good = findGoodPoints(grid_curr, spts)
            num_good = np.sum(grid_good)

            if num_good < 4:
                M = None
                break

            M = generateNewBestFit(ideal_grid, grid_next, grid_good)
            if M is None or abs(M[0, 0] / M[1, 1]) > 15:
                M = None
                break

        if M is not None and num_good > curr_num_good:
            curr_num_good = num_good
            curr_grid_next = grid_next
            curr_grid_good = grid_good
            curr_M = M

        if num_good > max_pts_needed:
            break

    if curr_num_good > min_pts_needed:
        final_ideal_grid = getIdentityGrid(16) - 7
        return curr_M, final_ideal_grid, curr_grid_next, curr_grid_good, spts

    return None, None, None, None, None