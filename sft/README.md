# SFT

Supervised fine-tuning of Qwen2.5-VL on RSC scenario data, run through
[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

## Data

`data/dataset_info.json` registers the dataset as `rsc-scenario` in ShareGPT
format. Place the training file next to it as `data/sft_sharegpt.jsonl`, one
JSON object per line:

```json
{
  "id": "rsc_sft_433806_629745",
  "images": ["/path/to/COCO_train2014_000000433806.jpg"],
  "conversations": [
    {"from": "user", "value": "<image>\nYou are given an image and a referring scenario description...\nReferring scenario description:\n<scenario>\n"},
    {"from": "assistant", "value": "<think>...</think>\n<answer>{\"target_object\": \"baseball bat\", \"bbox\": [282, 96, 41, 233]}</answer>"}
  ]
}
```

The number of image paths in `images` must match the number of `<image>` tokens
in the user turn. Any extra keys (e.g. a `meta` block) are ignored by
LLaMA-Factory and are convenient for bookkeeping.

The user turn uses the `think_answer` template — the same text as template 0 in
[`../grpo/prepare_grpo_data.py`](../grpo/prepare_grpo_data.py), so the GRPO
stages start from a format the policy already knows.

## Training

Set `dataset_dir` in the config to the absolute path of this directory's
`data/`, then launch from the root of your LLaMA-Factory checkout (the
`deepspeed: examples/deepspeed/ds_z3_config.json` path is inside that repo):

```bash
llamafactory-cli train /path/to/RSC/sft/qwen25vl_7b_sft.yaml
```

The vision tower and the multimodal projector are frozen: only the language
model is updated, which keeps perception intact while adapting reasoning and
output format. 5 epochs, LR 5e-6 cosine with 15% warmup, effective batch 8
(2 per device x 4 accumulation).

`qwen25vl_3b_sft.yaml` is the same recipe on Qwen2.5-VL-3B.

## Output

Checkpoints land in `saves/qwen25vl-7b-rsc-sft/`. The checkpoint used to
initialise GRPO stage 1 is step 2340; pass it as `INIT_CHECKPOINT` there.
