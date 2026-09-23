"""Object-category label normalisation.

Shared by the data builder (which stores the accepted names for each sample)
and by the category reward (which matches predictions against them), so the two
can never drift apart.
"""

import re
from typing import List

_WORD_RE = re.compile(r"[a-z0-9]+")

# Words that look plural but are not.
_INVARIANT_PLURALS = {"glasses", "scissors", "pants"}


def normalize_tokens(s: str) -> List[str]:
    """Lower-case, strip punctuation, and apply a tiny singulariser."""
    out = []
    for t in _WORD_RE.findall((s or "").lower()):
        if t.endswith("ies") and len(t) > 3:
            t = t[:-3] + "y"
        elif t.endswith("sses") and len(t) > 4:
            t = t[:-2]
        elif t.endswith("s") and len(t) > 3 and t not in _INVARIANT_PLURALS:
            t = t[:-1]
        out.append(t)
    return out


def normalize_label(s: str) -> str:
    return " ".join(normalize_tokens(s)).strip()


# Kept as a separate name because the data builder and the reward talk about
# "canonical" names; there is currently no alias table on top of normalisation.
canonicalize_label = normalize_label


def jaccard(a: List[str], b: List[str]) -> float:
    A, B = set(a), set(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)
