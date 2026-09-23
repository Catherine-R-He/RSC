#!/usr/bin/env python3
"""Re-filter result files to a subset of ann_ids and recompute the metrics.

Useful for scoring an existing run against a narrower set (a cleaned split, a
tag slice) without re-running the model, since ``eval_grounding.py`` writes
every per-sample record it scored.

    python tools/recalc_metrics.py \\
        --subset ../data/rsc_id_test.jsonl \\
        --results results/*.json \\
        --outdir results_cleaned/

One difference from ``eval_grounding.py``: category accuracy is counted over
records with a non-empty ``pred_canonical``, whereas the evaluator counts
records where the model named anything at all. They differ only when a
prediction normalises to the empty string.
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rsc_eval.metrics import center_l2_norm, iou_xywh, l1_bbox_norm  # noqa: E402

METRIC_KEYS = ("mIoU", "Acc@0.5", "Acc@0.7", "CenterL2(norm)", "L1_bbox(norm)",
               "CatAcc(strict)", "CatAcc(loose)")


def recompute(details: List[dict], dim_map: Dict[int, Tuple[int, int]]) -> dict:
    ious, cdist, l1err = [], [], []
    acc05 = acc07 = 0
    cat_total = cat_correct_strict = cat_correct_loose = 0

    for d in details:
        cat = d.get("cat", {})
        if cat and cat.get("pred_canonical"):
            cat_total += 1
            cat_correct_strict += int(bool(cat.get("match_strict")))
            cat_correct_loose += int(bool(cat.get("match_loose")))

        gt, pred = d.get("gt_xywh"), d.get("pred_xywh")
        if pred and gt:
            iou = d.get("IoU")
            if iou is None:
                iou = iou_xywh(gt, pred)
            ious.append(iou)
            if iou >= 0.5:
                acc05 += 1
            if iou >= 0.7:
                acc07 += 1
            dims = dim_map.get(d["ann_id"])
            if dims:
                W, H = dims
                cdist.append(center_l2_norm(gt, pred, H, W))
                l1err.append(l1_bbox_norm(gt, pred, H, W))

    n, nc = max(1, len(ious)), max(1, len(cdist))
    return {
        "samples": len(ious),
        "mIoU": round(sum(ious) / n, 4),
        "Acc@0.5": round(acc05 / n, 4),
        "Acc@0.7": round(acc07 / n, 4),
        "CenterL2(norm)": round(sum(cdist) / nc, 4),
        "L1_bbox(norm)": round(sum(l1err) / nc, 4),
        "CatSamples": cat_total,
        "CatAcc(strict)": round(cat_correct_strict / max(1, cat_total), 4),
        "CatAcc(loose)": round(cat_correct_loose / max(1, cat_total), 4),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", required=True,
                    help="JSONL whose ann_ids (and W/H) define the kept records")
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    print(f"Loading subset from {args.subset} ...")
    records = []
    with open(args.subset, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    kept_ids = {r["ann_id"] for r in records}
    dim_map = {r["ann_id"]: (int(r["W"]), int(r["H"])) for r in records if "W" in r and "H" in r}
    print(f"  {len(kept_ids)} ann_ids, {len(dim_map)} with W/H")

    os.makedirs(args.outdir, exist_ok=True)
    summary = []

    for result_path in args.results:
        name = os.path.basename(result_path)
        print(f"\nProcessing {name} ...")
        with open(result_path, encoding="utf-8") as f:
            data = json.load(f)

        orig_report = data.get("report", {})
        orig_details = data.get("details", [])
        filtered = [d for d in orig_details if d.get("ann_id") in kept_ids]
        print(f"  {len(orig_details)} → {len(filtered)} records "
              f"({len(orig_details) - len(filtered)} removed)")

        new_report = {k: orig_report[k] for k in ("provider", "model", "vocab") if k in orig_report}
        new_report["cleaned_from"] = name
        new_report["orig_samples"] = orig_report.get("samples", len(orig_details))
        new_report["kept_ann_ids"] = len(filtered)
        new_report.update(recompute(filtered, dim_map))

        out_path = os.path.join(args.outdir, name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"report": new_report, "details": filtered}, f, indent=2, ensure_ascii=False)
        print(f"  Saved → {out_path}")

        print(f"  {'Metric':<20} {'Original':>10} {'Recomputed':>12}")
        print(f"  {'-' * 44}")
        for k in METRIC_KEYS:
            print(f"  {k:<20} {str(orig_report.get(k, '—')):>10} {str(new_report.get(k, '—')):>12}")
        summary.append((orig_report, new_report))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for orig, new in summary:
        print(f"  {orig.get('model', new['cleaned_from'])}")
        print(f"    orig:       Acc@0.5={orig.get('Acc@0.5','?')}  mIoU={orig.get('mIoU','?')}  "
              f"samples={orig.get('samples','?')}")
        print(f"    recomputed: Acc@0.5={new.get('Acc@0.5','?')}  mIoU={new.get('mIoU','?')}  "
              f"samples={new.get('kept_ann_ids','?')}")
    print()


if __name__ == "__main__":
    main()
