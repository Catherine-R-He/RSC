#!/usr/bin/env bash
# Build the GRPO parquet splits.
#
# The SFT-template ratio is annealed across stages: stage 1 mostly reuses the
# phrasing the SFT model saw, stage 2 mixes in the other templates, and the
# validation split uses only the SFT template.
set -euo pipefail

cd "$(dirname "$0")/.."

ANNO_DIR=${ANNO_DIR:-../data/train}
COCO_ROOT=${COCO_ROOT:?set COCO_ROOT to the COCO root (holding 2014/)}
OUT_DIR=${OUT_DIR:-data}
SEED=${SEED:-1337}

python prepare_grpo_data.py \
  --refs-json "$ANNO_DIR/grpo_stage1.jsonl" \
  --image-root "$COCO_ROOT" \
  --out-dir "$OUT_DIR" \
  --split stage1 \
  --multi-templates \
  --sft-ratio 0.7 \
  --seed "$SEED"

python prepare_grpo_data.py \
  --refs-json "$ANNO_DIR/grpo_stage2.jsonl" \
  --image-root "$COCO_ROOT" \
  --out-dir "$OUT_DIR" \
  --split stage2 \
  --multi-templates \
  --sft-ratio 0.4 \
  --seed "$SEED"

python prepare_grpo_data.py \
  --refs-json "$ANNO_DIR/grpo_val.jsonl" \
  --image-root "$COCO_ROOT" \
  --out-dir "$OUT_DIR" \
  --split val \
  --multi-templates \
  --sft-ratio 1 \
  --seed "$SEED"
