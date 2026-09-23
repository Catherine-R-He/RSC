#!/usr/bin/env bash
# Build the GRPO parquet splits.
#
# The SFT-template ratio is annealed across stages: stage 1 mostly reuses the
# phrasing the SFT model saw, stage 2 mixes in the other templates, and the
# validation split is fully mixed so the eval never flatters a single template.
set -euo pipefail

cd "$(dirname "$0")/.."

ANNO_DIR=${ANNO_DIR:?set ANNO_DIR to the directory holding stage1/stage2/val jsonl}
OUT_DIR=${OUT_DIR:-data}
SEED=${SEED:-1337}

python prepare_grpo_data.py \
  --refs-json "$ANNO_DIR/stage1.jsonl" \
  --out-dir "$OUT_DIR" \
  --split stage1 \
  --multi-templates \
  --sft-ratio 0.7 \
  --seed "$SEED"

python prepare_grpo_data.py \
  --refs-json "$ANNO_DIR/stage2.jsonl" \
  --out-dir "$OUT_DIR" \
  --split stage2 \
  --multi-templates \
  --sft-ratio 0.4 \
  --seed "$SEED"

python prepare_grpo_data.py \
  --refs-json "$ANNO_DIR/val.jsonl" \
  --out-dir "$OUT_DIR" \
  --split val \
  --multi-templates \
  --sft-ratio 1 \
  --seed "$SEED"
