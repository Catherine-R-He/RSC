"""Turn a decoded completion into (thought, object_name, bbox).

Deliberately forgiving: the point of the eval is to measure grounding, not to
punish a stray code fence. We accept code fences, `//` and `/* */` comments,
smart quotes, single-quoted JSON, `{"output": {...}}` wrappers, xyxy or dict
boxes, and normalised or pixel coordinates.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

_TAG_THINK_RE = re.compile(r"<\s*think\s*>([\s\S]*?)<\s*/\s*think\s*>", re.I)
_TAG_ANSWER_RE = re.compile(r"<\s*answer\s*>([\s\S]*?)<\s*/\s*answer\s*>", re.I)


# ---------------------------------------------------------------------------
# Tolerant JSON recovery
# ---------------------------------------------------------------------------

def _strip_code_fences(s: str) -> str:
    return re.sub(r"```(?:json)?\s*|\s*```", "", s)


def _strip_js_comments(s: str) -> str:
    s = re.sub(r"//.*?$", "", s, flags=re.MULTILINE)
    return re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)


def _normalize_quotes(s: str) -> str:
    return s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")


def _brace_candidates(s: str) -> List[str]:
    """Every top-level ``{...}`` substring, found with a brace stack."""
    candidates, depth, start = [], 0, None
    for i, ch in enumerate(s):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(s[start:i + 1])
                start = None
    return candidates


def coerce_json(text: str) -> Optional[Dict[str, Any]]:
    """Parse the last JSON-looking block in ``text``."""
    s = _strip_js_comments(_normalize_quotes(_strip_code_fences(text)))
    for blob in reversed(_brace_candidates(s)):
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            try:  # single -> double quote fallback
                return json.loads(re.sub(r"(?<!\\)'", '"', blob))
            except json.JSONDecodeError:
                continue
    return None


# ---------------------------------------------------------------------------
# Box extraction
# ---------------------------------------------------------------------------

def extract_bbox(obj: Dict[str, Any]) -> Optional[List[float]]:
    """Pull a 4-tuple box out of a parsed payload, accepting several spellings."""
    out = obj.get("output", obj) if isinstance(obj, dict) else None
    bbox = None
    if isinstance(out, dict):
        bbox = out.get("bbox")
        if bbox is None:
            if all(k in out for k in ("x", "y", "w", "h")):
                bbox = [out["x"], out["y"], out["w"], out["h"]]
            elif isinstance(out.get("xywh"), (list, tuple)) and len(out["xywh"]) == 4:
                bbox = list(out["xywh"])
            elif isinstance(out.get("xyxy"), (list, tuple)) and len(out["xyxy"]) == 4:
                x1, y1, x2, y2 = out["xyxy"]
                bbox = [x1, y1, x2 - x1, y2 - y1]
    if isinstance(bbox, dict) and all(k in bbox for k in ("x", "y", "w", "h")):
        bbox = [bbox["x"], bbox["y"], bbox["w"], bbox["h"]]
    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
        return None
    return [float(v) for v in bbox]


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def finalize_bbox_xywh(bbox_xywh: List[float], H: int, W: int,
                       expect_norm: bool) -> Optional[List[int]]:
    """Scale (if normalised), clip to the image, round. None if degenerate.

    When ``expect_norm`` is off we still rescale a box whose width and height
    are both <= 1, since that can only be a normalised box.
    """
    x, y, w, h = bbox_xywh
    if expect_norm or (w <= 1.0 and h <= 1.0 and 0.0 <= w and 0.0 <= h):
        x, y, w, h = x * W, y * H, w * W, h * H
    w, h = max(0.0, w), max(0.0, h)
    x = _clip(x, 0.0, float(W))
    y = _clip(y, 0.0, float(H))
    w = _clip(w, 0.0, float(W) - x)
    h = _clip(h, 0.0, float(H) - y)
    xi, yi, wi, hi = int(round(x)), int(round(y)), int(round(w)), int(round(h))
    if wi == 0 or hi == 0:
        return None
    return [xi, yi, wi, hi]


# ---------------------------------------------------------------------------
# Full completion -> fields
# ---------------------------------------------------------------------------

def extract_think_and_answer(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (last ``<think>`` body, last ``<answer>`` body)."""
    thoughts = _TAG_THINK_RE.findall(text)
    answers = _TAG_ANSWER_RE.findall(text)
    return (thoughts[-1].strip() if thoughts else None,
            answers[-1].strip() if answers else None)


def parse_model_output(decoded_text: str, H: int, W: int,
                       expect_norm: bool = False
                       ) -> Tuple[Optional[str], Optional[str], Optional[List[int]]]:
    """Return ``(thought, object_name, bbox_xywh_int)``; fields may be None.

    JSON inside ``<answer>...</answer>`` wins; otherwise we fall back to the
    last JSON object anywhere in the text, so JSON-only models still score.
    """
    thought_tag, answer_inner = extract_think_and_answer(decoded_text)

    jd = coerce_json(answer_inner) if answer_inner else None
    if jd is None:
        jd = coerce_json(decoded_text)
    if jd is None:
        return thought_tag, None, None

    payload = jd.get("output", jd) if isinstance(jd, dict) else {}
    if not isinstance(payload, dict):
        payload = {}

    thought = thought_tag or (
        jd.get("thought") or jd.get("reasoning") or jd.get("rationale")
        if isinstance(jd, dict) else None
    )

    object_name = (
        payload.get("object_name")
        or payload.get("target_object")
        or (jd.get("object_name") if isinstance(jd, dict) else None)
        or (jd.get("target_object") if isinstance(jd, dict) else None)
    )
    if isinstance(object_name, (list, dict)):
        object_name = str(object_name)

    bbox_f = extract_bbox(jd)
    bbox_xywh = finalize_bbox_xywh(bbox_f, H, W, expect_norm) if bbox_f else None

    return thought, object_name, bbox_xywh
