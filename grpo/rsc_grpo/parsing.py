"""Extract the predicted JSON payload from a model completion.

The target format is::

    <think> ... </think>
    <answer>{"target_object": "...", "bbox": [x, y, w, h]}</answer>

Parsing is deliberately tolerant: json5 if available, a balanced-brace fallback
when the tags are missing, and a handful of key aliases.
"""

import json
import re
from typing import Any, Dict, List, Tuple

try:
    import json5  # tolerant JSON parsing
    HAS_JSON5 = True
except Exception:
    HAS_JSON5 = False

ANSWER_OPEN = "<answer>"
ANSWER_CLOSE = "</answer>"
ANSWER_RE = re.compile(r"<answer>([\s\S]*?)</answer>", re.IGNORECASE)

_BBOX_KEYS = ("bbox", "bbox_xywh", "box")
_OBJECT_KEYS = ("target_object", "object", "category", "label")
_WRAPPER_KEYS = ("output", "result", "prediction", "data")


def first_json_object(text: str):
    """Return the first balanced ``{...}`` block in ``text``, or None."""
    if not text:
        return None
    stack = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if stack == 0:
                start = i
            stack += 1
        elif ch == "}":
            if stack > 0:
                stack -= 1
                if stack == 0 and start != -1:
                    return text[start:i + 1]
    return None


def to_dict(maybe_json):
    """Coerce str/bytes/dict into a dict via json, then json5."""
    if isinstance(maybe_json, dict):
        return maybe_json
    if isinstance(maybe_json, (bytes, bytearray)):
        maybe_json = maybe_json.decode("utf-8", errors="ignore")
    if isinstance(maybe_json, str):
        try:
            return json.loads(maybe_json)
        except Exception:
            if HAS_JSON5:
                try:
                    return json5.loads(maybe_json)
                except Exception:
                    return None
            return None
    return None


def completion_text(c) -> str:
    """TRL usually hands us strings, but tolerate chat-style list/dict forms."""
    if isinstance(c, str):
        return c
    if isinstance(c, list) and c:
        msg = c[0]
        if isinstance(msg, dict):
            return str(msg.get("content", "")) or ""
        return str(msg)
    if isinstance(c, dict):
        return str(c.get("content", "")) or ""
    return str(c or "")


def _unwrap(data: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten common wrappers such as ``{"output": {...}}``."""
    if not isinstance(data, dict):
        return {}
    for key in _WRAPPER_KEYS:
        if key in data:
            inner = data[key]
            if isinstance(inner, str):
                inner = to_dict(inner) or {}
            if isinstance(inner, dict):
                return inner
    return data


def extract_answer_payload(text: str):
    """Return the raw string inside ``<answer>...</answer>``, or None."""
    if not text:
        return None
    m = ANSWER_RE.search(text)
    return m.group(1).strip() if m else None


def parse_answer_json(text: str):
    """Prefer JSON inside the answer tags; fall back to the first ``{...}``."""
    if not text:
        return None
    payload = extract_answer_payload(text)
    if payload:
        d = to_dict(payload)
        if isinstance(d, dict):
            return d
    json_block = first_json_object(text)
    if json_block:
        d = to_dict(json_block)
        if isinstance(d, dict):
            return d
    return None


def has_answer_tags(text: str) -> bool:
    return ANSWER_OPEN in text and ANSWER_CLOSE in text


def safe_parse_completion(text: str) -> Tuple[List[int], str]:
    """Parse a completion into ``(bbox_xywh, target_object)``.

    Either element is None when it is missing or malformed.
    """
    if not text:
        return None, None

    data = parse_answer_json(text)
    if not isinstance(data, dict):
        return None, None
    data = _unwrap(data)

    bbox = next((data[k] for k in _BBOX_KEYS if data.get(k)), None)
    obj = next((data[k] for k in _OBJECT_KEYS if data.get(k)), None)

    if isinstance(bbox, list) and len(bbox) == 4:
        try:
            bbox = [int(round(float(x))) for x in bbox]
        except Exception:
            bbox = None
    else:
        bbox = None

    if obj is not None and not isinstance(obj, str):
        obj = str(obj)

    return bbox, obj
