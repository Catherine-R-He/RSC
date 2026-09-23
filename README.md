# RSC

Training code for RSC — referring-scenario grounding with Qwen2.5-VL.

Given an image and a *scenario* description (a situated, multi-sentence
reference rather than a short referring expression), the model reasons inside
`<think>...</think>` and emits a single JSON object inside `<answer>...</answer>`:

```json
{"target_object": "baseball bat", "bbox": [282, 96, 41, 233]}
```

`bbox` is `[x, y, w, h]` in pixels.

## Pipeline

The released model is produced in three stages, each initialised from the
previous one:

| Stage | Code | Init from | Purpose |
| --- | --- | --- | --- |
| SFT | [`sft/`](sft) | `Qwen/Qwen2.5-VL-7B-Instruct` | Teach the think/answer format and basic scenario grounding |
| GRPO stage 1 | [`grpo/train_stage1.py`](grpo/train_stage1.py) | SFT checkpoint (step 2340) | Lock in the output contract, coarse localisation |
| GRPO stage 2 | [`grpo/train_stage2.py`](grpo/train_stage2.py) | stage-1 checkpoint (step 400) | Precision refinement on the harder split |

The final model is the stage-2 checkpoint at step 100.

## Evaluation data

| Split | File | Records | Vocabulary |
| --- | --- | --- | --- |
| In-domain | [`data/rsc_id_test.jsonl`](data/rsc_id_test.jsonl) | 4,038 | COCO |
| Out-of-domain | [`data/rsc_ood_test.jsonl`](data/rsc_ood_test.jsonl) | 3,247 | LVIS |

Image paths are relative to a COCO root; see [`data/README.md`](data/README.md).

Each stage sees a different mix of prompt templates. Template 0 is the phrasing
the SFT model was trained on; its sampling probability drops from 0.7 in stage 1
to 0.4 in stage 2, so the policy is progressively forced to generalise beyond
the SFT phrasing. The validation split is fully template-mixed.

## Layout

```
data/                      the two evaluation splits + the script that built them
sft/                       LLaMA-Factory configs for the SFT stage
grpo/
  prepare_grpo_data.py     scenario JSONL -> GRPO parquet splits
  train_stage1.py          stage-1 GRPO entry point
  train_stage2.py          stage-2 GRPO entry point
  rsc_grpo/                shared library (parsing, geometry, rewards, callbacks)
  configs/                 accelerate + DeepSpeed ZeRO-3 configs
  scripts/                 launch scripts
eval/
  eval_grounding.py        in-domain and out-of-domain evaluation
  rsc_eval/                shared library (parsing, metrics, labels, aliases)
  aliases/                 LVIS category alias data
  tools/                   re-score an existing result file on a subset
  scripts/                 launch scripts
```

## Setup

The two stages use different, mutually incompatible environments (LLaMA-Factory
pins an older `transformers`/`trl` than TRL's GRPO trainer needs), so keep them
in separate conda environments:

```bash
# SFT
conda create -n rsc-sft python=3.10 && conda activate rsc-sft
# clone LLaMA-Factory, then `pip install -e .` there
pip install -r sft/requirements.txt

# GRPO (also covers evaluation)
conda create -n rsc-grpo python=3.10 && conda activate rsc-grpo
pip install -r grpo/requirements.txt
```

Evaluation needs only a subset of the GRPO environment; `eval/requirements.txt`
lists it for a standalone install.

Both stages log to Weights & Biases (`wandb login`, or set `WANDB_MODE=offline`).

## Running

See [`sft/README.md`](sft/README.md) and [`grpo/README.md`](grpo/README.md) for
the per-stage training instructions, and [`eval/README.md`](eval/README.md) for
evaluation:

```bash
export COCO_ROOT=/path/to/COCO
MODEL=/path/to/qwen25vl-grpo-bbox-stage2/checkpoint-100 eval/scripts/eval_id.sh
MODEL=/path/to/qwen25vl-grpo-bbox-stage2/checkpoint-100 eval/scripts/eval_ood.sh
```

## Citation

TODO: add the paper reference and model/dataset links before release.
