# Evaluation

One script for both splits. `--vocab` selects the category vocabulary used both
in the prompt and for alias matching.

```bash
export COCO_ROOT=/path/to/COCO

# In-domain (4,038 samples)
MODEL=/path/to/qwen25vl-grpo-bbox-stage2/checkpoint-100 scripts/eval_id.sh

# Out-of-domain (3,247 samples)
MODEL=/path/to/qwen25vl-grpo-bbox-stage2/checkpoint-100 scripts/eval_ood.sh
```

Or directly:

```bash
python eval_grounding.py \
  --grounding-json ../data/rsc_id_test.jsonl \
  --vocab coco \
  --model /path/to/checkpoint \
  --image-root "$COCO_ROOT" \
  --out-dir results
```

## Options

| Flag | Default | Meaning |
| --- | --- | --- |
| `--grounding-json` | — | Evaluation JSONL (required) |
| `--model` | — | Checkpoint directory or HF id (required) |
| `--vocab` | `coco` | `coco` for the in-domain split, `lvis` for out-of-domain |
| `--base-model` | `Qwen/Qwen2.5-VL-7B-Instruct` | Where the architecture and generation config come from |
| `--processor` | `--base-model` | Processor path |
| `--image-root` | `$COCO_ROOT` | Prepended to relative `image_path` values |
| `--include-risky-aliases` | off | COCO only: also fold the broad aliases |
| `--expect-norm` | off | Model emits normalised boxes |
| `--max-samples` | all | Truncate for a quick check |
| `--cuda` / `--attn-impl` | `0` / `flash_attention_2` | Device and attention kernel |
| `--out-dir` / `--out` | `./results` / derived | Where the result JSON lands |

A GRPO checkpoint directory can carry a `config.json` that sends transformers
down the seq2seq path, so the architecture and generation configs are loaded
from `--base-model` and only the weights come from `--model`. Decoding uses the
base model's generation config with `temperature=0.01`, `top_p=1.0` — the
released numbers depend on this, so changing it changes the results.

## Metrics

The report holds:

| Metric | Meaning |
| --- | --- |
| `mIoU` | Mean IoU over samples where a box was produced |
| `Acc@0.5`, `Acc@0.7` | Fraction above the IoU threshold |
| `CenterL2(norm)` | Centre distance over the image diagonal |
| `L1_bbox(norm)` | Mean absolute coordinate error, normalised |
| `CatAcc(strict)` | Predicted category canonicalises to the ground-truth one |
| `CatAcc(loose)` | Every ground-truth token appears in the prediction (so "blurry traffic city light" counts for "traffic light") |
| `samples` / `CatSamples` | Denominators: samples with a box / with a named object |

Alongside the report, every scored sample is written out (prediction, GT,
thought, tags), so results can be re-scored or sliced without re-running the
model:

```bash
python tools/recalc_metrics.py \
  --subset ../data/rsc_id_test.jsonl \
  --results results/*.json \
  --outdir results_subset/
```

## Category aliases

Predicted names are matched through an alias table before comparison, so
`sofa` counts for `couch`.

* **COCO** — hand-curated, inline in `rsc_eval/aliases.py`. A second
  `LOOSE_ALIASES_RISKY` table holds broader folds (`chair` <- `seat`,
  `car` <- `vehicle`); it is off by default and the reported numbers use the
  safe set only.
* **LVIS** — `aliases/lvis_synonym_to_canon_safe.json`, 3,628 entries generated
  from the official LVIS `synonyms` fields. `aliases/lvis_alias_collisions.json`
  records the aliases that were dropped for mapping to more than one category.

## Layout

```
eval_grounding.py      entry point for both splits
rsc_eval/
  parsing.py           <answer> / JSON recovery from completions
  metrics.py           IoU, centre distance, box L1
  labels.py            category normalisation and matching
  aliases.py           COCO alias tables + vocabulary loading
  data.py              JSONL reading, image path resolution
  model.py             Qwen2.5-VL wrapper and the prompt
aliases/               LVIS alias data
tools/recalc_metrics.py  re-score an existing result file on a subset
scripts/               launch wrappers
```
