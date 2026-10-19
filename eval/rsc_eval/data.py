"""Reading the RSC evaluation JSONL."""

import json
import os
from typing import Any, Dict, List, Optional


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
    """Ground-truth box as xywh floats."""
    b = rec.get("bbox")
    if isinstance(b, (list, tuple)) and len(b) == 4:
        return [float(v) for v in b]
    return None


def get_scenario(rec: Dict[str, Any]) -> Optional[str]:
    s = rec.get("scenario")
    if isinstance(s, str) and s.strip():
        return s.strip()
    return None
