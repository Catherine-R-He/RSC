# Data

Two evaluation splits of referring-*scenario* annotations over COCO images.

| File | Records | Vocabulary | Images |
| --- | --- | --- | --- |
| `rsc_id_test.jsonl` | 4,038 | COCO (in-domain) | COCO `train2014` |
| `rsc_ood_test.jsonl` | 3,247 | LVIS (out-of-domain) | COCO `train2017` (2,431) and `val2017` (816) |

## Images

`image_path` is relative to a COCO root, so point the evaluator at yours:

```
$COCO_ROOT/
  2014/train2014/COCO_train2014_000000045840.jpg
  train2017/000000335323.jpg
  val2017/000000012345.jpg
```

```bash
export COCO_ROOT=/path/to/COCO
```

The two splits reference 5,664 unique images (~700 MB) across three COCO
releases whose full archives total ~32 GB, so `download_images.py` fetches just
those:

```bash
python download_images.py --image-root "$COCO_ROOT"
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
  "split": "val",
  "scenario": "As a graphic designer selecting a background for a couple's adventure poster, you need the large, shiny black vehicle beneath them, not the people or the grass.",
  "referring_expression": "The shiny black, chrome-accented vehicle supporting both people and stretching across the image's lower center.",
  "thought": "The scenario asks for the object that is large, shiny black, and supports both people, ...",
  "anno_object_name": "black cruiser-style two-wheeled vehicle",
  "accecptable_names": ["motorcycle", "motor bike", "motorbike", "bike"],
  "anno_object_attributes": {"color": "...", "material": "...", "shape": "...", "texture": "...",
                             "size": "...", "position": "...", "affordance": "...", "relation": "..."},
  "tags": {"U": "U1", "C": "C1", "S": "L", "O": "O0", "P": "P0"},
  "U": null, "C": null, "S": null, "O": null, "P": null
}
```

* `bbox` is ground truth in pixel `[x, y, w, h]`.
* `scenario` is the evaluation input; `referring_expression` is the shorter
  reference and `thought` the annotator's rationale, kept for analysis.
* `category_name` is the ground-truth category the prediction is scored
  against; `accecptable_names` are the annotator's alternatives (the evaluator
  scores against the alias tables in `../eval/aliases`, not this field).
* `tags` are the difficulty axes used for slicing results: uniqueness (`U`),
  clutter (`C`), size (`S`), occlusion (`O`), position (`P`). The top-level
  `U`/`C`/`S`/`O`/`P` keys are an older flat copy and are usually null.

## Rebuilding

`make_release_data.py` produced these from the annotation pipeline's output: it
rewrites `image_path` to be relative and drops pipeline bookkeeping
(`response_json` / `response_json_raw`, whose useful fields are already at the
top level, and `pred_bbox_xywh` / `iou` / `category_match`, which are the
annotator's own self-check rather than a model prediction).

```bash
python make_release_data.py \
  --in /path/to/splits_v2_test_gpt41_v10.filtered.clean.quality.jsonl \
  --out rsc_id_test.jsonl \
  --image-root /scratch/data/COCO
```
