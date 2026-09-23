#!/usr/bin/env bash
# Stage-1 GRPO, initialised from the SFT checkpoint.
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export INIT_CHECKPOINT=${INIT_CHECKPOINT:?set INIT_CHECKPOINT to the SFT checkpoint directory}
export RSC_DATA_DIR=${RSC_DATA_DIR:-data}

accelerate launch \
  --config_file configs/accelerate_zero3.yaml \
  --deepspeed_config_file configs/ds_zero3.json \
  train_stage1.py
