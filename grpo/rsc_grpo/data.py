"""Loading the GRPO parquet splits written by ``prepare_grpo_data.py``.

Images are stored as paths and opened lazily per batch, so a split costs almost
nothing to keep in memory. Set ``IMG_PREFIX_FROM`` / ``IMG_PREFIX_TO`` when the
parquet was built on a machine whose image root differs from this one.
"""

import os

from datasets import load_dataset
from PIL import Image

IMG_PREFIX_FROM = os.environ.get("IMG_PREFIX_FROM", "")
IMG_PREFIX_TO = os.environ.get("IMG_PREFIX_TO", "")


def rewrite_path(p: str) -> str:
    if IMG_PREFIX_FROM and IMG_PREFIX_TO and isinstance(p, str) and p.startswith(IMG_PREFIX_FROM):
        return IMG_PREFIX_TO + p[len(IMG_PREFIX_FROM):]
    return p


def _sanitize_record(rec):
    """Drop stray keys from prompt parts.

    Image parts must be ``{'type': 'image', 'image': <path or PIL>}`` and text
    parts ``{'type': 'text', 'text': str}``; anything else confuses the
    processor's chat template.
    """
    def fix_parts(parts):
        out = []
        for p in (parts or []):
            t = p.get("type")
            if t == "image":
                img = p.get("image")
                if isinstance(img, str):
                    img = rewrite_path(img)
                out.append({"type": "image", "image": img})
            elif t == "text":
                out.append({"type": "text", "text": p.get("text") or ""})
        return out

    pr = rec.get("prompt")
    if isinstance(pr, list):
        rec["prompt"] = [
            {"role": m.get("role", "user"), "content": fix_parts(m.get("content"))}
            for m in pr
        ]
    return rec


def _lazy_open_images(rec):
    """Turn image paths into PIL images, after the schema is known clean."""
    pr = rec.get("prompt")
    if isinstance(pr, list):
        for m in pr:
            parts = m.get("content") or []
            for i, p in enumerate(parts):
                if p.get("type") == "image" and isinstance(p.get("image"), str):
                    try:
                        with Image.open(p["image"]) as im:
                            parts[i] = {"type": "image", "image": im.convert("RGB").copy()}
                    except Exception:
                        # Leave the path in place; the processor will raise for
                        # this sample only if it also fails to open it.
                        pass
    return rec


def load_split(path: str):
    ds = load_dataset("parquet", data_files=path, split="train")
    ds = ds.map(
        _sanitize_record,
        batched=False,
        load_from_cache_file=False,  # never reuse a stale prompt schema
        desc="Sanitizing prompt schema",
    )
    ds = ds.with_transform(_lazy_open_images)
    return ds.with_format("python")


def sanitize_messages(messages):
    """Same cleanup as ``_sanitize_record``, for messages seen at runtime."""
    out = []
    for m in (messages or []):
        parts = []
        for p in (m.get("content") or []):
            if p.get("type") == "image":
                parts.append({"type": "image", "image": p.get("image")})
            elif p.get("type") == "text":
                parts.append({"type": "text", "text": p.get("text") or ""})
        out.append({"role": m.get("role", "user"), "content": parts})
    return out
