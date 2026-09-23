#!/usr/bin/env python3
"""Build the GRPO parquet splits from RSC scenario annotations.

Input is JSONL, one referring-scenario sample per line, with at least::

    {"image_path": "...", "H": 480, "W": 640,
     "bbox_xywh": [x, y, w, h], "category_name": "dog",
     "scenario": "the dog that ...",
     "acceptable_names": ["dog", "puppy"]}       # optional

Output is a parquet file holding the chat-format prompt plus the columns the
reward functions need (W, H, gt_bbox_xywh, the accepted category names, and the
template id used).

Each sample is rendered with one of eight prompt templates. Template 0 is the
one the SFT model was trained on; ``--sft-ratio`` controls how often it is
drawn, and we lower it across stages (0.7 -> 0.4 -> 0.1) so the policy is
progressively forced to generalise beyond the SFT phrasing instead of
memorising it.
"""

import argparse
import json
import os
import random

import datasets

from rsc_grpo.labels import canonicalize_label, normalize_label
from rsc_grpo.parsing import ANSWER_CLOSE, ANSWER_OPEN

REGEX_ANSWER_EXTRACT = r"<answer>([\s\S]*?)</answer>"

# ---------------------------------------------------------------------------
# Accepted category names
# ---------------------------------------------------------------------------

_ALIAS_KEYS = ("acceptable_names", "accecptable_names", "aliases", "synonyms")


def accepted_names(rec: dict) -> list:
    """Collect every name that should count as correct for this sample."""
    names = []
    for k in _ALIAS_KEYS:
        v = rec.get(k)
        if isinstance(v, list):
            names += [str(x) for x in v]
    for nest in ("response_json", "response_json_raw"):
        j = rec.get(nest)
        if isinstance(j, dict):
            for k in _ALIAS_KEYS:
                v = j.get(k)
                if isinstance(v, list):
                    names += [str(x) for x in v]
    gt = rec.get("category_name")
    if isinstance(gt, str) and gt.strip():
        names.append(gt.strip())

    seen, out = set(), []
    for n in names:
        key = n.strip().lower()
        if key and key not in seen:
            out.append(n.strip())
            seen.add(key)
    return out


# ---------------------------------------------------------------------------
# Prompt templates - all of them ask for <think>...</think> then <answer>JSON</answer>
# ---------------------------------------------------------------------------

def _tmpl_classic(H, W, scenario):
    return (
        "You are given an image and a referring scenario description.\n"
        "First, think step by step inside <think>...</think>.\n"
        "Then output ONE JSON object inside <answer>...</answer> exactly:\n"
        "<answer>{\n"
        '  "target_object": "<canonical object category>",\n'
        '  "bbox": [x, y, w, h]\n'
        "}</answer>\n"
        "Rules:\n"
        "- The only content outside <answer> should be <think>…</think>.\n"
        "- \"bbox\" are 4 INTEGERS [x,y,w,h], xywh in pixels, clipped to bounds.\n"
        f"Image size: Height={int(H)}, Width={int(W)}\n"
        "Referring scenario description:\n"
        f"{scenario}\n"
    )


def _tmpl_json_only(H, W, scenario):
    return (
        "Task: detect the referred object.\n"
        "Think inside <think>…</think>, then put JSON ONLY inside <answer>…</answer>:\n"
        "<answer>{\n"
        '  "target_object": "<canonical object category>",\n'
        '  "bbox": [x, y, w, h]\n'
        "}</answer>\n"
        "Constraints:\n"
        f"- bbox are 4 INTEGERS (xywh in pixels), 0<=x<x+w<={int(W)}, 0<=y<y+h<={int(H)}\n"
        "- No prose, no markdown, no extra keys anywhere.\n"
        f"Scenario:\n{scenario}\n"
    )


def _tmpl_schema(H, W, scenario):
    return (
        "Output must match this schema, with JSON only inside <answer>…</answer>:\n"
        'type ObjectBox = {"target_object": string, "bbox": [int,int,int,int]}\n'
        "<think>List the cues and verify constraints.</think>\n"
        "<answer>{\n"
        '  "target_object": "<canonical object category>",\n'
        '  "bbox": [x, y, w, h]\n'
        "}</answer>\n"
        f"Image size: H={int(H)}, W={int(W)}\n"
        f"Scenario:\n{scenario}\n"
    )


def _tmpl_tool(H, W, scenario):
    return (
        "Return arguments for a function as pure JSON **inside <answer>…</answer>**.\n"
        "function predict_bbox(target_object: string, bbox: [int,int,int,int])\n"
        "<think>Reason about object and bounds.</think>\n"
        "<answer>{\n"
        '  "target_object": "<canonical object category>",\n'
        '  "bbox": [x, y, w, h]\n'
        "}</answer>\n"
        f"Rules: bbox ints; xywh in pixels; within image (H={int(H)}, W={int(W)}).\n"
        f"Scenario:\n{scenario}\n"
    )


def _tmpl_concise(H, W, scenario):
    return (
        "Be concise. Think in <think>…</think>. Put ONLY JSON in <answer>…</answer>.\n"
        "<think>Identify object + final box.</think>\n"
        "<answer>{\n"
        '  "target_object": "<canonical object category>",\n'
        '  "bbox": [x, y, w, h]\n'
        "}</answer>\n"
        f"(H={int(H)}, W={int(W)})\n"
        f"Scenario:\n{scenario}\n"
    )


def _tmpl_validator(H, W, scenario):
    return (
        "Validate before answering. Use <think> for checks; JSON goes in <answer> only.\n"
        f"Checks: integers; 0<=x<x+w<={int(W)}; 0<=y<y+h<={int(H)}; non-zero area; canonical category.\n"
        "<think>Run the checklist, then finalize.</think>\n"
        "<answer>{\n"
        '  "target_object": "<canonical object category>",\n'
        '  "bbox": [x, y, w, h]\n'
        "}</answer>\n"
        f"Scenario:\n{scenario}\n"
    )


def _tmpl_checklist(H, W, scenario):
    return (
        "Checklist (use in <think>):\n"
        f"- Integers xywh; clipped to W={int(W)}, H={int(H)}\n"
        "- One object, one JSON; canonical category\n"
        "<think>Apply the checklist.</think>\n"
        "<answer>{\n"
        '  "target_object": "<canonical object category>",\n'
        '  "bbox": [x, y, w, h]\n'
        "}</answer>\n"
        f"Scenario:\n{scenario}\n"
    )


def _tmpl_coordinate(H, W, scenario):
    return (
        "Coordinate reminder: origin (0,0) top-left; bbox is xywh.\n"
        "Think in <think>; answer as JSON ONLY inside <answer>.\n"
        "<think>Derive xywh; clip to bounds.</think>\n"
        "<answer>{\n"
        '  "target_object": "<canonical object category>",\n'
        '  "bbox": [x, y, w, h]\n'
        "}</answer>\n"
        f"Bounds: 0<=x<x+w<={int(W)}, 0<=y<y+h<={int(H)}\n"
        f"Scenario:\n{scenario}\n"
    )


PROMPT_TEMPLATES = (
    _tmpl_classic,
    _tmpl_json_only,
    _tmpl_schema,
    _tmpl_tool,
    _tmpl_concise,
    _tmpl_validator,
    _tmpl_checklist,
    _tmpl_coordinate,
)

TEMPLATE_NAMES = (
    "think_answer",   # id 0 - the template the SFT model was trained on
    "json_only",
    "schema_hint",
    "tool_style",
    "concise",
    "validator",
    "checklist",
    "coordinate_hint",
)


# ---------------------------------------------------------------------------
# Record -> TRL sample
# ---------------------------------------------------------------------------

def _weighted_choice(rng: random.Random, probs):
    r, s = rng.random(), 0.0
    for i, p in enumerate(probs):
        s += p
        if r <= s:
            return i
    return len(probs) - 1  # numerical safety


def _user_message(parts):
    return [{"role": "user", "content": parts}]


def make_map_fn(split: str, seed: int, multi_templates: bool,
                template_probs, single_template_id: int):
    rng = random.Random(seed)
    K = len(PROMPT_TEMPLATES)

    if template_probs is not None:
        assert len(template_probs) == K, f"expected {K} template probabilities"
        total = sum(template_probs)
        assert total > 0, "template probabilities must sum to > 0"
        probs = [p / total for p in template_probs]
    else:
        probs = [1.0 / K] * K

    def process_fn(ex, idx):
        H, W = int(ex["H"]), int(ex["W"])
        scenario = ex.get("scenario") or (ex.get("response_json", {}) or {}).get("scenario") or ""
        image_path = ex["image_path"]

        x, y, w, h = [int(round(v)) for v in (ex.get("bbox_xywh") or ex.get("bbox"))]
        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))
        w = max(0, min(w, W - x))
        h = max(0, min(h, H - y))

        t_id = _weighted_choice(rng, probs) if multi_templates else single_template_id
        prompt_text = PROMPT_TEMPLATES[t_id](H, W, scenario)

        paths = image_path if isinstance(image_path, list) else [image_path]
        parts = [{"type": "image", "image": str(p)} for p in paths]
        parts.append({"type": "text", "text": prompt_text})

        allowed_raw = accepted_names(ex)
        return {
            "prompt": _user_message(parts),
            "W": W, "H": H,
            "gt_bbox_xywh": [x, y, w, h],
            "category_name": ex.get("category_name", ""),
            "cat_allowed_raw": allowed_raw,
            "cat_allowed_canonical": sorted({canonicalize_label(n) for n in allowed_raw if n}),
            "cat_allowed_norm": sorted({normalize_label(n) for n in allowed_raw if n}),
            "data_source": "rsc",
            "split": split,
            "index": idx,
            "image_path": image_path,
            "image_id": ex.get("image_id", ""),
            "ann_id": ex.get("ann_id", ""),
            "scenario": scenario,
            "stop": [ANSWER_CLOSE],
            "regex_answer_extract": REGEX_ANSWER_EXTRACT,
            "answer_open": ANSWER_OPEN,
            "answer_close": ANSWER_CLOSE,
            "template_id": t_id,
            "template_name": TEMPLATE_NAMES[t_id],
        }

    return process_fn


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return rows


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refs-json", required=True, help="input JSONL of scenario samples")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split", default="train",
                    choices=["train", "stage1", "stage2", "stage3", "val", "test"])
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--multi-templates", action="store_true",
                    help="sample among the 8 prompt templates instead of using one")
    ap.add_argument("--template", default="think_answer", choices=list(TEMPLATE_NAMES),
                    help="template to use when --multi-templates is not set")
    ap.add_argument("--sft-ratio", type=float, default=None,
                    help="probability of the SFT template (id 0); the rest share 1-ratio uniformly")
    ap.add_argument("--template-probs", type=str, default=None,
                    help="comma-separated probabilities for all 8 templates; overrides --sft-ratio")
    return ap.parse_args()


def main():
    args = parse_args()

    template_probs = None
    if args.template_probs:
        template_probs = [float(x) for x in args.template_probs.split(",")]
    elif args.sft_ratio is not None:
        r = max(0.0, min(1.0, float(args.sft_ratio)))
        K = len(PROMPT_TEMPLATES)
        template_probs = [r] + [(1.0 - r) / (K - 1)] * (K - 1)

    os.makedirs(args.out_dir, exist_ok=True)
    refs = load_jsonl(args.refs_json)
    print(f"Processing {len(refs)} samples for split='{args.split}' "
          f"(multi_templates={args.multi_templates})")

    if template_probs is not None:
        total = sum(template_probs)
        print("Template probability mix →",
              ", ".join(f"{TEMPLATE_NAMES[i]}:{p / total:.3f}" for i, p in enumerate(template_probs)))

    ds = datasets.Dataset.from_list(refs).map(
        function=make_map_fn(args.split, args.seed, args.multi_templates,
                             template_probs, TEMPLATE_NAMES.index(args.template)),
        with_indices=True,
        desc=f"Converting {args.split} → TRL GRPO parquet",
    )

    from collections import Counter
    hist = Counter(ds["template_id"])
    tot = sum(hist.values())
    print("Realized template histogram →",
          ", ".join(f"{TEMPLATE_NAMES[k]}:{hist[k]} ({hist[k] / tot:.2%})" for k in sorted(hist)))

    out_path = os.path.join(args.out_dir, f"{args.split}.parquet")
    ds.to_parquet(out_path)
    print(f"✓ Wrote {out_path}")
    print(f"Columns: {ds.column_names}")


if __name__ == "__main__":
    main()
