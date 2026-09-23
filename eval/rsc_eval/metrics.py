"""Localisation metrics. All boxes are pixel-space ``[x, y, w, h]``."""

import math
from typing import List


def iou_xywh(a: List[float], b: List[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter + 1e-6
    return float(inter / union)


def center_l2_norm(a: List[float], b: List[float], H: int, W: int) -> float:
    """Centre distance, normalised by the image diagonal."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    acx, acy = ax + aw / 2.0, ay + ah / 2.0
    bcx, bcy = bx + bw / 2.0, by + bh / 2.0
    return math.hypot(acx - bcx, acy - bcy) / (math.hypot(W, H) + 1e-6)


def l1_bbox_norm(a: List[float], b: List[float], H: int, W: int) -> float:
    """Mean absolute error of the four box coordinates in normalised units."""
    an = [a[0] / W, a[1] / H, a[2] / W, a[3] / H]
    bn = [b[0] / W, b[1] / H, b[2] / W, b[3] / H]
    return float(sum(abs(x - y) for x, y in zip(an, bn)) / 4.0)
