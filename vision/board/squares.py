import numpy as np


def _to_int(x):
    return int(round(float(x)))


def build_9_lines(best_lines):
    """
    best_lines: 7 belső vonal (np.array)
    visszaad: 9 határvonal (külsőkkel együtt)
    """
    best_lines = np.asarray(best_lines, dtype=np.float32).reshape(-1)
    if len(best_lines) != 7:
        raise ValueError(f"best_lines must have length 7, got {len(best_lines)}")

    d = best_lines[1] - best_lines[0]
    lines9 = np.concatenate([[best_lines[0] - d], best_lines, [best_lines[-1] + d]])
    return lines9, float(d)


def extract_squares_from_warp(img_warp, best_x, best_y, inner_pad_ratio=0.12):
    """
    img_warp: warpolt (felülnézeti) szürke vagy színes kép
    best_x, best_y: 7 belső vonal x/y irányban
    inner_pad_ratio: a mező széléből levágott arány

    Returns:
      bbox_warp: 8x8 lista (x0,y0,x1,y1) warp koordinátában
      centers_warp: 8x8 lista (cx,cy) warp koordinátában
      lines: (x_lines9, y_lines9)
    """
    h, w = img_warp.shape[:2]

    x_lines, _ = build_9_lines(best_x)
    y_lines, _ = build_9_lines(best_y)

    bbox_warp = [[None for _ in range(8)] for _ in range(8)]
    centers_warp = [[None for _ in range(8)] for _ in range(8)]

    for r in range(8):
        for c in range(8):
            x0f, x1f = x_lines[c], x_lines[c + 1]
            y0f, y1f = y_lines[r], y_lines[r + 1]

            x0f, x1f = (x0f, x1f) if x0f <= x1f else (x1f, x0f)
            y0f, y1f = (y0f, y1f) if y0f <= y1f else (y1f, y0f)

            padx = inner_pad_ratio * (x1f - x0f)
            pady = inner_pad_ratio * (y1f - y0f)

            x0 = _to_int(x0f + padx)
            x1 = _to_int(x1f - padx)
            y0 = _to_int(y0f + pady)
            y1 = _to_int(y1f - pady)

            x0 = max(0, min(w - 1, x0))
            x1 = max(0, min(w, x1))
            y0 = max(0, min(h - 1, y0))
            y1 = max(0, min(h, y1))

            if x1 - x0 < 4 or y1 - y0 < 4:
                x0 = max(0, min(w - 1, _to_int(x0f)))
                x1 = max(0, min(w, _to_int(x1f)))
                y0 = max(0, min(h - 1, _to_int(y0f)))
                y1 = max(0, min(h, _to_int(y1f)))

            bbox_warp[r][c] = (x0, y0, x1, y1)

            cx = 0.5 * (x0f + x1f)
            cy = 0.5 * (y0f + y1f)
            centers_warp[r][c] = (float(cx), float(cy))

    return bbox_warp, centers_warp, (x_lines, y_lines)