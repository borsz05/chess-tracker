from __future__ import annotations

import cv2


CLASS_COLORS_BGR = {
    0: (120, 120, 120),   # empty
    1: (80, 220, 80),     # white
    2: (80, 80, 220),     # black
}


def draw_square_class_dots(frame, centers_img, raw_labels, radius: int = 8) -> None:
    if centers_img is None or raw_labels is None:
        return

    for row in range(8):
        for col in range(8):
            cx, cy = centers_img[row][col]
            label = int(raw_labels[row][col])
            color = CLASS_COLORS_BGR.get(label, (0, 255, 255))

            center = (int(round(cx)), int(round(cy)))
            cv2.circle(frame, center, radius + 2, (0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(frame, center, radius, color, thickness=-1, lineType=cv2.LINE_AA)
