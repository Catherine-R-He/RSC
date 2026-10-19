# Data

Referring-*scenario* annotations over COCO images: two evaluation splits, plus
the training files the released model was trained on.

| File | Records | Use | Images |
| --- | --- | --- | --- |
| `rsc_id_test.jsonl` | 4,038 | In-domain test (COCO vocabulary) | COCO `train2014` |
| `rsc_ood_test.jsonl` | 3,247 | Out-of-domain test (LVIS vocabulary) | COCO `train2017` (2,431) and `val2017` (816) |
| `train/sft.jsonl` | 30,000 | SFT (ShareGPT format) | COCO `train2014` and `val2014` |
| `train/grpo_stage1.jsonl` | 4,000 | GRPO stage 1 | COCO `train2014` and `val2014` |
| `train/grpo_stage2.jsonl` | 3,500 | GRPO stage 2 | COCO `train2014` and `val2014` |
| `train/grpo_val.jsonl` | 5,000 | GRPO validation | COCO `train2014` |

No training record shares an annotation with either test split.

## Images

`image_path` is relative to a COCO root, so point the evaluator at yours:

```
$COCO_ROOT/
  2014/train2014/COCO_train2014_000000045840.jpg
  2014/val2014/COCO_val2014_000000000294.jpg
  train2017/000000335323.jpg
  val2017/000000012345.jpg
```

```bash
export COCO_ROOT=/path/to/COCO
```

The two test splits reference 5,664 unique images (~700 MB) across three COCO
releases whose full archives total ~32 GB, so `download_images.py` fetches just
those. Add `--with-train` to also fetch the training images (33,276 in total,
~5.5 GB):

```bash
python download_images.py --image-root "$COCO_ROOT"               # test splits only
python download_images.py --image-root "$COCO_ROOT" --with-train  # test + training
```

It skips files that already exist, so it is safe to re-run after an interruption
and safe to point at an existing COCO tree — it only fills in what is missing.

## Record format

```json
{
  "image_id": 45840,
  "ann_id": 149077,
  "category_id": 4,
  "category_name": "motorcycle",
  "bbox": [104.0, 136.0, 426.0, 233.0],
  "image_path": "2014/train2014/COCO_train2014_000000045840.jpg",
  "W": 640, "H": 426,
  "source_split": "val",
  "scenario": "As a graphic designer selecting a background for a couple's adventure poster, you need the large, shiny black vehicle beneath them, not the people or the grass.",
  "referring_expression": "The shiny black, chrome-accented vehicle supporting both people and stretching across the image's lower center.",
  "thought": "The scenario asks for the object that is large, shiny black, and supports both people, ...",
  "anno_object_name": "black cruiser-style two-wheeled vehicle",
  "acceptable_names": ["motorcycle", "motor bike", "motorbike", "bike"],
  "anno_object_attributes": {"color": "...", "material": "...", "shape": "...", "texture": "...",
                             "size": "...", "position": "...", "affordance": "...", "relation": "..."},
  "tags": {"U": "U1", "C": "C1", "S": "L", "O": "O0", "P": "P0"}
}
```

* `bbox` is ground truth in pixel `[x, y, w, h]`.
* `scenario` is the evaluation input; `referring_expression` is the shorter
  reference and `thought` the annotator's rationale, kept for analysis.
* `category_name` is the ground-truth category the prediction is scored
  against; `acceptable_names` are the annotator's alternatives (the evaluator
  scores against the alias tables in `../eval/aliases`, not this field).
* `tags` are the difficulty axes used for slicing results: uniqueness (`U`),
  clutter (`C`), size (`S`), occlusion (`O`), position (`P`).
* `source_split` is the split of the upstream annotation pool the object was
  drawn from, kept for provenance. It is not the RSC split: every record in
  `rsc_id_test.jsonl` is a test record.

The GRPO training files use the same record format, without `source_split`.
`train/sft.jsonl` is in ShareGPT format; see [`../sft/README.md`](../sft/README.md).

