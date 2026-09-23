"""Reading the RSC evaluation JSONL."""

import json
import os
from typing import Any, Dict, List, Optional, Tuple


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def resolve_image_path(rec: Dict[str, Any], image_root: Optional[str]) -> str:
    """Released records store ``image_path`` relative to the COCO root."""
    p = rec["image_path"]
    return os.path.join(image_root, p) if image_root and not os.path.isabs(p) else p


def get_gt_bbox(rec: Dict[str, Any]) -> Optional[List[float]]:
    """Ground-truth box as xywh floats.

    The ``response_json`` fallbacks are for raw annotation-pipeline output; the
    released files carry the box at the top level.
    """
    for k in ("bbox_xywh", "bbox", "gt_xywh"):
        b = rec.get(k)
        if isinstance(b, (list, tuple)) and len(b) == 4:
            return [float(v) for v in b]

    rj = rec.get("response_json") or rec.get("response_json_raw")
    if isinstance(rj, dict):
        for k in ("bbox_xywh", "bbox"):
            b = rj.get(k)
            if isinstance(b, (list, tuple)) and len(b) == 4:
                return [float(v) for v in b]
        bn = rj.get("bbox_normalized")
        if isinstance(bn, (list, tuple)) and len(bn) == 4 and rec.get("W") and rec.get("H"):
            W, H = float(rec["W"]), float(rec["H"])
            return [bn[0] * W, bn[1] * H, bn[2] * W, bn[3] * H]
    return None


def get_scenario(rec: Dict[str, Any]) -> Optional[str]:
    s = rec.get("scenario")
    if isinstance(s, str) and s.strip():
        return s.strip()
    rj = rec.get("response_json") or rec.get("response_json_raw")
    if isinstance(rj, dict):
        s = rj.get("scenario")
        if isinstance(s, str) and s.strip():
            return s.strip()
    return None
