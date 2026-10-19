#!/usr/bin/env python3
"""Evaluate a Qwen2.5-VL checkpoint on RSC scenario grounding.

One script for both splits; ``--vocab`` picks the category vocabulary used for
the prompt and for alias matching:

    --vocab coco   in-domain  (data/rsc_id_test.jsonl,  4,038 samples)
    --vocab lvis   out-of-domain (data/rsc_ood_test.jsonl, 3,247 samples)

Reports mIoU, Acc@0.5, Acc@0.7, normalised centre distance and box L1, plus
strict and loose category accuracy. Per-sample records are written alongside
the report so metrics can be recomputed or sliced without re-running the model.
"""

import argparse
import json
import os
from typing import Any, Dict, Optional

import torch
from PIL import Image
from tqdm import tqdm

from rsc_eval.aliases import load_synonym_map
from rsc_eval.data import get_gt_bbox, get_scenario, load_jsonl, resolve_image_path
from rsc_eval.labels import canonicalize_label, loose_contains_match, normalize_label
from rsc_eval.metrics import center_l2_norm, iou_xywh, l1_bbox_norm
from rsc_eval.model import ModelCfg, QwenVL

VOCAB_LABELS = {"coco": "COCO", "lvis": "LVIS"}


def evaluate(grounding_json: str,
             model_path: str,
             processor_path: Optional[str] = None,
             base_model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
             vocab: str = "coco",
             include_risky_aliases: bool = False,
             image_root: Optional[str] = None,
             cuda: int = 0,
             max_samples: int = -1,
             attn_impl: str = "flash_attention_2",
             expect_norm: bool = False) -> Dict[str, Any]:

    device = f"cuda:{cuda}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    synonym_map = load_synonym_map(vocab, include_risky=include_risky_aliases)
    qwen = QwenVL(ModelCfg(
        model_path=model_path,
        processor_path=processor_path,
        base_model_id=base_model,
        device=device,
        attn_impl=attn_impl,
        vocab_label=VOCAB_LABELS[vocab],
    ))

    items = load_jsonl(grounding_json)
    if max_samples > 0:
        items = items[:max_samples]

    ious, cdist, l1err = [], [], []
    acc05 = acc07 = 0
    cat_total = cat_correct_strict = cat_correct_loose = 0
    details = []

    for rec in tqdm(items, desc="Evaluating"):
        img_path = resolve_image_path(rec, image_root)
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        W, H = img.size

        gt = get_gt_bbox(rec)
        if gt is None:
            continue
        scenario = get_scenario(rec) or ""

        thought, pred_name, pred = qwen.predict_box(img, scenario, H, W, expect_norm=expect_norm)
        iou = iou_xywh(gt, pred) if pred else 0.0

        # ---- Category metrics (scored only when the model named something) ----
        gt_canon = canonicalize_label(rec.get("category_name") or "", synonym_map)
        pred_canon = canonicalize_label(pred_name or "", synonym_map)
        pred_norm = normalize_label(pred_name or "")

        cat_scored = bool(pred_name)
        cat_exact = (pred_canon == gt_canon) if cat_scored else False
        cat_loose = loose_contains_match(gt_canon, pred_norm) if cat_scored else False
        if cat_scored:
            cat_total += 1
            cat_correct_strict += int(cat_exact)
            cat_correct_loose += int(cat_loose)

        details.append({
            "image_id": rec.get("image_id"),
            "ann_id": rec.get("ann_id"),
            "image_path": rec.get("image_path"),
            "category_name": rec.get("category_name"),
            "tags": rec.get("tags"),
            "anno_object": rec.get("anno_object_name"),
            "scenario_used": scenario,
            "thought": thought,
            "pred_object": pred_name,
            "gt_xywh": [round(v, 2) for v in gt],
            "pred_xywh": [round(v, 2) for v in pred] if pred else None,
            "IoU": round(iou, 4),
            "cat": {
                "gt_canonical": gt_canon,
                "pred_canonical": pred_canon,
                "match_strict": bool(cat_exact),
                "match_loose": bool(cat_loose),
            },
        })

        if pred:
            ious.append(iou)
            if iou >= 0.5:
                acc05 += 1
            if iou >= 0.7:
                acc07 += 1
            cdist.append(center_l2_norm(gt, pred, H, W))
            l1err.append(l1_bbox_norm(gt, pred, H, W))

    n = max(1, len(ious))
    report = {
        "model": model_path,
        "vocab": vocab,
        "samples": len(ious),
        "mIoU": round(sum(ious) / n, 4),
        "Acc@0.5": round(acc05 / n, 4),
        "Acc@0.7": round(acc07 / n, 4),
        "CenterL2(norm)": round(sum(cdist) / max(1, len(cdist)), 4),
        "L1_bbox(norm)": round(sum(l1err) / max(1, len(l1err)), 4),
        "CatSamples": int(cat_total),
        "CatAcc(strict)": round(cat_correct_strict / max(1, cat_total), 4),
        "CatAcc(loose)": round(cat_correct_loose / max(1, cat_total), 4),
    }
    return {"report": report, "details": details}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grounding-json", required=True, help="evaluation JSONL")
    ap.add_argument("--model", required=True, help="HF id or local checkpoint dir")
    ap.add_argument("--vocab", default="coco", choices=sorted(VOCAB_LABELS),
                    help="coco for the in-domain split, lvis for out-of-domain")
    ap.add_argument("--processor", default=None,
                    help="processor path (defaults to --base-model)")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-VL-7B-Instruct",
                    help="source of the architecture and generation config")
    ap.add_argument("--image-root", default=os.environ.get("COCO_ROOT"),
                    help="prepended to relative image_path values (env: COCO_ROOT)")
    ap.add_argument("--include-risky-aliases", action="store_true",
                    help="COCO only: also fold the broad aliases (looser category scoring)")
    ap.add_argument("--cuda", type=int, default=0)
    ap.add_argument("--max-samples", type=int, default=-1)
    ap.add_argument("--attn-impl", default="flash_attention_2",
                    choices=["flash_attention_2", "eager"])
    ap.add_argument("--expect-norm", action="store_true",
                    help="model emits normalised boxes")
    ap.add_argument("--out-dir", default="./results")
    ap.add_argument("--out", default=None, help="output filename (default: derived)")
    return ap.parse_args()


def main():
    args = parse_args()

    res = evaluate(
        grounding_json=args.grounding_json,
        model_path=args.model,
        processor_path=args.processor,
        base_model=args.base_model,
        vocab=args.vocab,
        include_risky_aliases=args.include_risky_aliases,
        image_root=args.image_root,
        cuda=args.cuda,
        max_samples=args.max_samples,
        attn_impl=args.attn_impl,
        expect_norm=args.expect_norm,
    )
    print(json.dumps(res["report"], indent=2))

    os.makedirs(args.out_dir, exist_ok=True)
    name = args.out or (
        os.path.splitext(os.path.basename(args.grounding_json))[0]
        + f".eval_{os.path.basename(args.model).replace(os.sep, '_')}.json"
    )
    out_path = os.path.join(args.out_dir, name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"✓ Wrote {out_path}")


if __name__ == "__main__":
    main()
