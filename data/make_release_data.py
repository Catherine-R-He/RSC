#!/usr/bin/env python3
"""Turn the annotation pipeline's output into the released evaluation JSONL.

Two changes:

1. ``image_path`` becomes relative to the COCO root, so the files are portable.
   Evaluation resolves them against ``--image-root``.
2. Pipeline bookkeeping is dropped. ``response_json`` / ``response_json_raw``
   hold the raw annotator response whose useful fields are already promoted to
   the top level, and ``pred_bbox_xywh`` / ``iou`` / ``category_match`` are the
   annotator's own self-check, not a model prediction. Keeping them would
   triple the file size and invite mistaking the self-check for a baseline.

Usage:
    python make_release_data.py \\
        --in  /path/to/splits_v2_test_gpt41_v10.filtered.clean.quality.jsonl \\
        --out rsc_id_test.jsonl \\
        --image-root /scratch/data/COCO
"""

import argparse
import json
import os

# Kept in this order; a key missing from a record is simply skipped.
KEEP_FIELDS = [
    "image_id", "ann_id", "category_id", "category_name", "bbox",
    "image_path", "W", "H", "split",
    "scenario", "referring_expression", "thought",
    "anno_object_name", "accecptable_names", "anno_object_attributes",
    "tags", "U", "C", "S", "O", "P",
]


def relativize(path: str, image_root: str) -> str:
    root = image_root.rstrip("/") + "/"
    if not path.startswith(root):
        raise ValueError(f"{path!r} is not under --image-root {image_root!r}")
    return path[len(root):]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--image-root", required=True,
                    help="prefix stripped from image_path (e.g. /scratch/data/COCO)")
    args = ap.parse_args()

    n = 0
    with open(args.inp, encoding="utf-8") as fin, open(args.out, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec["image_path"] = relativize(rec["image_path"], args.image_root)
            slim = {k: rec[k] for k in KEEP_FIELDS if k in rec}
            fout.write(json.dumps(slim, ensure_ascii=False) + "\n")
            n += 1

    print(f"✓ {args.out}: {n} records, {os.path.getsize(args.out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
