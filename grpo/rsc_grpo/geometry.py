"""Bounding-box geometry helpers.

All boxes are pixel-space ``[x, y, w, h]`` (xywh) unless a function name says
otherwise.
"""

import math
from typing import List, Sequence


def xywh_to_xyxy(b: Sequence[float]) -> List[float]:
    x, y, w, h = b
    return [x, y, x + w, y + h]


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter) / denom if denom > 0 else 0.0


def clip_xywh(b: Sequence[float], W: int, H: int) -> List[int]:
    """Clip to the image and enforce at least 1 px of width/height (area > 0)."""
    x, y, w, h = b
    x = max(0, min(int(round(x)), W - 1))
    y = max(0, min(int(round(y)), H - 1))
    w = max(1, min(int(round(w)), W - x))
    h = max(1, min(int(round(h)), H - y))
    return [x, y, w, h]


def guess_and_norm_bbox_xywh(b: Sequence[float], W: int, H: int) -> List[int]:
    """Coerce a predicted 4-tuple to a valid xywh box.

    Models occasionally emit xyxy instead of xywh, so we pick whichever reading
    is geometrically plausible and fall back to clipping as xywh.
    """
    x, y, a, b2 = [float(v) for v in b]

    # Prefer xywh: (x, y) inside the image, (a, b2) plausible sizes that fit.
    if (0 <= x < W) and (0 <= y < H) and (1 <= a <= W) and (1 <= b2 <= H) \
       and (x + a <= W) and (y + b2 <= H):
        return clip_xywh([x, y, a, b2], W, H)

    # Otherwise try xyxy: (a, b2) look like (x2, y2) strictly past (x, y).
    if (0 <= x < W) and (0 <= y < H) and (0 <= a <= W) and (0 <= b2 <= H) \
       and (a > x) and (b2 > y):
        return clip_xywh([x, y, a - x, b2 - y], W, H)

    return clip_xywh([x, y, a, b2], W, H)


def center_xywh(b: Sequence[float]):
    x, y, w, h = b
    return (x + w / 2.0, y + h / 2.0)


def center_bonus(pred, gt, W: int, H: int, sigma: float = 0.20) -> float:
    """Gaussian reward on centre distance, normalised by the image diagonal.

    ``sigma`` is a fraction of the diagonal; the return value is in [0, 0.2].
    """
    (cx, cy), (gx, gy) = center_xywh(pred), center_xywh(gt)
    diag = math.hypot(W, H)
    d = math.hypot(cx - gx, cy - gy) / max(1.0, diag)
    return math.exp(-(d ** 2) / (2 * (sigma ** 2))) * 0.2
