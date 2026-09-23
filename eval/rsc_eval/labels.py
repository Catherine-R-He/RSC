"""Category-name normalisation and matching."""

import re
from typing import Optional

_WORD_RE = re.compile(r"[a-z0-9]+")

# Words that look plural but are not.
_INVARIANT_PLURALS = {"glasses", "scissors", "pants"}


def _singularize(w: str) -> str:
    if w.endswith("ies") and len(w) > 3:
        return w[:-3] + "y"        # batteries -> battery
    if w.endswith("sses") and len(w) > 4:
        return w[:-2]              # glasses -> glass (keeps 'ss')
    if w.endswith("s") and len(w) > 3 and w not in _INVARIANT_PLURALS:
        return w[:-1]              # cats -> cat
    return w


def normalize_label(s: Optional[str]) -> str:
    """Lower-case, drop punctuation, collapse spaces, singularise."""
    if not s:
        return ""
    return " ".join(_singularize(t) for t in _WORD_RE.findall(s.lower())).strip()


def canonicalize_label(s: Optional[str], synonym_map: dict) -> str:
    """Normalise, then fold through the alias map if it has a hit."""
    n = normalize_label(s)
    if not n:
        return ""
    if n in synonym_map:
        return synonym_map[n]
    joined = n.replace(" ", "")  # "hair drier" written as "hairdrier"
    if joined in synonym_map:
        return synonym_map[joined]
    return n


def loose_contains_match(gt_canon: str, pred_norm: str) -> bool:
    """True when every token of the canonical GT appears in the prediction.

    Lets a descriptive answer count, e.g. gt ``traffic light`` against
    ``blurry traffic city light``.
    """
    if not gt_canon or not pred_norm:
        return False
    return set(gt_canon.split()).issubset(set(pred_norm.split()))
