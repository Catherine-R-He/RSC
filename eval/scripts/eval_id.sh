#!/usr/bin/env bash
# In-domain evaluation (COCO vocabulary, 4,038 samples).
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL=${MODEL:?set MODEL to a checkpoint directory or HF model id}
COCO_ROOT=${COCO_ROOT:?set COCO_ROOT to the directory holding 2014/, train2017/, val2017/}
OUT_DIR=${OUT_DIR:-results}
CUDA=${CUDA:-0}

python eval_grounding.py \
  --grounding-json ../data/rsc_id_test.jsonl \
  --vocab coco \
  --model "$MODEL" \
  --image-root "$COCO_ROOT" \
  --out-dir "$OUT_DIR" \
  --cuda "$CUDA" \
  --attn-impl flash_attention_2
