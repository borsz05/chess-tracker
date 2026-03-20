import numpy as np


def compute_square_diffs(prev_warp, curr_warp, bbox_grid):
    """
    Megmondja mely mezők változtak leginkább.
    """
    diffs = []

    for r in range(8):
        for c in range(8):
            x0, y0, x1, y1 = bbox_grid[r][c]

            prev_sq = prev_warp[y0:y1, x0:x1]
            curr_sq = curr_warp[y0:y1, x0:x1]

            if prev_sq.size == 0 or curr_sq.size == 0:
                diff = 0
            else:
                d = np.abs(prev_sq.astype(np.int16) - curr_sq.astype(np.int16))
                diff = float(np.mean(d))

            diffs.append((diff, r, c))

    diffs.sort(reverse=True)
    return diffs