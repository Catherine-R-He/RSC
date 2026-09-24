# Beyond Referring Expressions: Scenario Comprehension Visual Grounding

Code and evaluation data for **RSC** (Referring Scenario Comprehension).

[Paper (arXiv:2604.02323)](https://arxiv.org/abs/2604.02323) ·
Ruozhen He, Nisarg A. Shah, Qihua Dong, Zilin Xiao, Jaywon Koo, Vicente Ordonez

## Overview

Classic referring-expression grounding gives the model a short phrase that
names the target. RSC instead gives a **scenario**: a situated description of
what someone is doing and why, from which the target has to be *inferred*.

> *As a graphic designer selecting a background for a couple's adventure
> poster, you need the large, shiny black vehicle beneath them, not the people
> or the grass.*

The model reasons inside `<think>...</think>` and emits one JSON object inside
`<answer>...</answer>`:

```json
{"target_object": "motorcycle", "bbox": [104, 136, 426, 233]}
```

`bbox` is `[x, y, w, h]` in pixels. Predictions are scored on localisation
(IoU) and on naming the right category.

## Repository layout

```
data/     the two evaluation splits, and the COCO image downloader
sft/      supervised fine-tuning configs (LLaMA-Factory)
grpo/     two-stage GRPO training (TRL)
eval/     in-domain and out-of-domain evaluation
```

Each directory has its own README with the details; this page is the path from
a clean machine to a number.

---

## 1. Setup

```bash
git clone https://github.com/Catherine-R-He/RSC.git
cd RSC
```

Training needs **two environments**: LLaMA-Factory pins an older
`transformers`/`trl` than TRL's GRPO trainer requires, so they cannot share one.
If you only want to run evaluation, the GRPO environment is enough.

**GRPO + evaluation**

```bash
conda create -n rsc-grpo python=3.10 && conda activate rsc-grpo
pip install -r grpo/requirements.txt
```

**SFT**

```bash
conda create -n rsc-sft python=3.10 && conda activate rsc-sft
git clone https://github.com/hiyouga/LLaMA-Factory.git
pip install -e LLaMA-Factory
pip install -r sft/requirements.txt
```

Both `requirements.txt` files pin the versions the released runs used. `torch`
and `flash-attn` must match your CUDA toolkit — install those first if your
setup is unusual, then the rest.

Training and evaluation log to Weights & Biases:

```bash
wandb login          # or: export WANDB_MODE=offline
```

## 2. Data

The annotations ship with this repository:

| Split | File | Records | Category vocabulary |
| --- | --- | --- | --- |
| In-domain | `data/rsc_id_test.jsonl` | 4,038 | COCO |
| Out-of-domain | `data/rsc_ood_test.jsonl` | 3,247 | LVIS |

The images do not — they are COCO. `image_path` is stored relative to a COCO
root, and the two splits together reference 5,664 images spread across three
COCO releases whose full archives are ~32 GB. The included downloader fetches
only what is needed (~700 MB):

```bash
export COCO_ROOT=/path/to/COCO
python data/download_images.py --image-root "$COCO_ROOT"
```

It skips files that already exist, so it is safe to re-run after an
interruption, and safe to point at an existing COCO tree — it will only fill in
what is missing. The expected layout is:

```
$COCO_ROOT/
  2014/train2014/COCO_train2014_000000045840.jpg
  train2017/000000335323.jpg
  val2017/000000012345.jpg
```

If your COCO copy is laid out differently, either symlink it into this shape or
pass `--image-root` pointing at the right level. See
[`data/README.md`](data/README.md) for the full record format.

## 3. Evaluation

```bash
conda activate rsc-grpo
export COCO_ROOT=/path/to/COCO

MODEL=/path/to/checkpoint eval/scripts/eval_id.sh    # in-domain,     4,038 samples
MODEL=/path/to/checkpoint eval/scripts/eval_ood.sh   # out-of-domain, 3,247 samples
```

`MODEL` is a local checkpoint directory or a Hugging Face model id. Results land
in `eval/results/` as a JSON file holding the summary report plus every
per-sample prediction. Or call the script directly:

```bash
python eval/eval_grounding.py \
  --grounding-json data/rsc_id_test.jsonl \
  --vocab coco \
  --model /path/to/checkpoint \
  --image-root "$COCO_ROOT" \
  --out-dir eval/results
```

Use `--vocab lvis` for the out-of-domain split; it switches both the category
vocabulary named in the prompt and the alias table used for scoring. Add
`--max-samples 50` for a quick smoke test before committing a GPU to the full
run. Full options are in [`eval/README.md`](eval/README.md).

**Metrics.** `mIoU`, `Acc@0.5` and `Acc@0.7` for localisation; `CenterL2(norm)`
and `L1_bbox(norm)` for how the misses fail; `CatAcc(strict)` and `CatAcc(loose)`
for naming, matched through a COCO or LVIS alias table so `sofa` counts for
`couch`.

**Results** for the three stages of the released pipeline:

| Model | ID mIoU | ID Acc@0.5 | ID CatAcc | OOD mIoU | OOD Acc@0.5 | OOD CatAcc |
| --- | --- | --- | --- | --- | --- | --- |
| SFT | 0.550 | 0.605 | 0.891 | 0.332 | 0.340 | 0.125 |
| + GRPO stage 1 | 0.557 | 0.610 | 0.944 | 0.369 | 0.366 | 0.190 |
| + GRPO stage 2 | 0.557 | 0.609 | 0.942 | **0.384** | **0.381** | 0.211 |

`CatAcc` is the strict variant. Sample counts differ by a handful between rows
because a sample only enters the localisation metrics when the model actually
produced a box.

## 4. Training

The released model is built in three stages, each initialised from the previous
one. Expect 8 GPUs; the GRPO stages run under DeepSpeed ZeRO-3.

### 4.1 SFT

Teaches the think/answer format and basic scenario grounding, starting from
`Qwen/Qwen2.5-VL-7B-Instruct`. The vision tower and multimodal projector are
frozen, so only the language model moves.

Put your ShareGPT-format training file at `sft/data/sft_sharegpt.jsonl` (format
in [`sft/README.md`](sft/README.md)), then run from your LLaMA-Factory checkout:

```bash
conda activate rsc-sft
llamafactory-cli train /path/to/RSC/sft/qwen25vl_7b_sft.yaml
```

5 epochs, LR 5e-6 cosine, effective batch 8. Checkpoints land in
`saves/qwen25vl-7b-rsc-sft/`; the released run used **step 2340**.
`sft/qwen25vl_3b_sft.yaml` is the same recipe on Qwen2.5-VL-3B.

### 4.2 GRPO data

Both RL stages read parquet splits built from scenario JSONL:

```bash
conda activate rsc-grpo
cd grpo
ANNO_DIR=/path/to/anno OUT_DIR=data scripts/prepare_data.sh
```

This writes `stage1.parquet`, `stage2.parquet` and `val.parquet`. Each sample is
rendered with one of eight prompt templates; the template the SFT model was
trained on is drawn 70% of the time in stage 1 and 40% in stage 2, so the policy
is progressively pushed off the phrasing it memorised.

### 4.3 GRPO stage 1

Locks in the output contract and coarse localisation, from the SFT checkpoint.

```bash
INIT_CHECKPOINT=/path/to/saves/qwen25vl-7b-rsc-sft/checkpoint-2340 \
  scripts/train_stage1.sh
```

5 epochs, LR 1e-6, 6 samples per prompt. The reward mixes IoU (0.75), category
(0.15), format (0.07) and structure (0.03) at fixed weights, and the
localisation reward pays nothing without well-formed `<answer>` tags. The
released run used **step 400**.

### 4.4 GRPO stage 2

Precision refinement on the harder split, from the stage-1 checkpoint.

```bash
INIT_CHECKPOINT=/path/to/qwen25vl-grpo-bbox-stage1/checkpoint-400 \
  scripts/train_stage2.sh
```

2 epochs, LR 2e-6, 12 samples per prompt and shorter completions. The reward mix
is now scheduled from format-friendly towards IoU-dominated over the first 60%
of training, the answer-tag gate is dropped, and out-of-bounds boxes are
penalised instead. The released run used **step 100** — this is the final model.

Both stages adapt the KL coefficient during training to hold the observed KL near
a target. Every path is an environment variable, so nothing needs editing for a
new machine; the full table is in [`grpo/README.md`](grpo/README.md).

### Checkpoint lineage

```
Qwen/Qwen2.5-VL-7B-Instruct
  └─ SFT               step 2340
       └─ GRPO stage 1 step 400
            └─ GRPO stage 2 step 100   ← released model
```

## Citation

```bibtex
@article{he2026beyond,
  title={Beyond Referring Expressions: Scenario Comprehension Visual Grounding},
  author={He, Ruozhen and Shah, Nisarg A and Dong, Qihua and Xiao, Zilin and Koo, Jaywon and Ordonez, Vicente},
  journal={arXiv preprint arXiv:2604.02323},
  year={2026}
}
```

## Acknowledgements

Built on [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL), with SFT through
[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) and GRPO through
[TRL](https://github.com/huggingface/trl). Images and category vocabularies come
from [COCO](https://cocodataset.org) and [LVIS](https://www.lvisdataset.org).
