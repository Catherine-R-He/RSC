# GRPO

Two-stage GRPO ([TRL](https://github.com/huggingface/trl)) on top of the SFT
checkpoint, with a composite reward over localisation, category, and output
format.

## 1. Build the data

Input is one JSONL file per split, one referring-scenario sample per line:

```json
{"image_path": "/path/to/image.jpg", "H": 480, "W": 640,
 "bbox_xywh": [282, 96, 41, 233], "category_name": "baseball bat",
 "scenario": "As a coach reviewing game footage, you want to ...",
 "acceptable_names": ["baseball bat", "bat"]}
```

`acceptable_names` is optional; when present, its entries also count as a
correct category (see the category reward below).

```bash
ANNO_DIR=/path/to/anno OUT_DIR=data scripts/prepare_data.sh
```

This writes `data/stage1.parquet`, `data/stage2.parquet`, and `data/val.parquet`.
Run `python prepare_grpo_data.py --help` for the per-split options.

## 2. Train

```bash
# Stage 1 - from the SFT checkpoint
INIT_CHECKPOINT=/path/to/saves/qwen25vl-7b-rsc-sft/checkpoint-2340 \
  scripts/train_stage1.sh

# Stage 2 - from the stage-1 checkpoint
INIT_CHECKPOINT=/path/to/qwen25vl-grpo-bbox-stage1/checkpoint-400 \
  scripts/train_stage2.sh
```

Both scripts launch 8 processes under DeepSpeed ZeRO-3. Paths and run names are
environment variables, so nothing needs editing for a new machine:

| Variable | Default | Meaning |
| --- | --- | --- |
| `BASE_MODEL` | `Qwen/Qwen2.5-VL-7B-Instruct` | Processor source; also the fallback if `INIT_CHECKPOINT` is not a directory |
| `INIT_CHECKPOINT` | — | Policy initialisation (required) |
| `RSC_DATA_DIR` | `data` | Where the parquet splits live |
| `TRAIN_PQ` / `VAL_PQ` | `$RSC_DATA_DIR/{stage,val}.parquet` | Override individual splits |
| `EXP_NAME` / `OUT_DIR` | `qwen25vl-grpo-bbox-stage{1,2}` | Run name / checkpoint directory |
| `DS_CONFIG` | `configs/ds_zero3.json` | DeepSpeed config |
| `IMG_PREFIX_FROM` / `IMG_PREFIX_TO` | unset | Rewrite image paths when the parquet was built elsewhere |

## Rewards

All four rewards are computed per completion, logged raw to W&B under
`rewards_raw/*`, then mixed:

| Reward | What it measures |
| --- | --- |
| `bbox_iou_soft_reward` | IoU with smooth sigmoid bonuses around the 0.5 / 0.7 thresholds, plus a small centre-distance term |
| `category_reward` | Predicted category against the sample's accepted names (1.0 canonical, 0.8 alias, 0.4–0.7 token overlap), halved when IoU < 0.3 |
| `format_reward` | ±1 for answer tags wrapping parseable JSON with both required fields |
| `structure_reward` | Partial credit: 0.25 for the tags, 0.75 for the required keys |

Weights:

| | IoU | Category | Format | Structure |
| --- | --- | --- | --- | --- |
| Stage 1 (fixed) | 0.75 | 0.15 | 0.07 | 0.03 |
| Stage 2 (start) | 0.55 | 0.25 | 0.12 | 0.08 |
| Stage 2 (end) | 0.75 | 0.20 | 0.04 | 0.01 |

Stage 2 interpolates linearly from start to end over the first 60% of training
(`RewardWeightScheduler`), so formatting is reinforced while the policy adapts
to the harder split and localisation takes over afterwards.

Stage-specific differences in the localisation reward:

* **Stage 1** scores 0 when the `<answer>` tags are missing — the output
  contract has to be learned first.
* **Stage 2** drops that gate (a well-localised box is no longer thrown away
  over formatting) and instead subtracts 0.05 when the predicted box had to be
  clipped or reinterpreted to become a valid in-bounds box.

Both stages adapt the KL coefficient `beta` during training
(`AdaptiveKLScheduler`) to hold the observed KL near a target (0.13 in stage 1,
0.15 in stage 2) within `[5e-4, 5e-2]`.

## Hyperparameters

| | Stage 1 | Stage 2 |
| --- | --- | --- |
| Epochs | 5 | 2 |
| LR (cosine, 5% warmup) | 1e-6 | 2e-6 |
| Generations per prompt | 6 | 12 |
| Generation batch size | 240 | 288 |
| Temperature / top-p | 0.8 / 0.95 | 0.9 / 0.92 |
| Max completion length | 256 | 160 |
| Per-device batch x accumulation | 1 x 32 | 1 x 32 |

## Layout

```
prepare_grpo_data.py   scenario JSONL -> GRPO parquet
train_stage1.py        stage-1 entry point (config at the top of the file)
train_stage2.py        stage-2 entry point
rsc_grpo/
  parsing.py           <answer> / JSON extraction from completions
  geometry.py          bbox conversion, IoU, clipping, centre bonus
  labels.py            category normalisation (shared with the data builder)
  data.py              parquet loading and lazy image opening
  model.py             processor and policy construction
  rewards.py           the four rewards and the weighting wrapper
  callbacks.py         adaptive KL and the stage-2 reward curriculum
configs/               accelerate + DeepSpeed ZeRO-3
scripts/               launch wrappers
```
