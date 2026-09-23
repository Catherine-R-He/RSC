"""Processor / model construction for Qwen2.5-VL GRPO runs."""

import os
from types import MethodType

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from .data import sanitize_messages


def build_processor(base_model: str, pixels: int = 512 * 512):
    processor = AutoProcessor.from_pretrained(
        base_model,
        padding_side="left",  # RL log-prob alignment prefers left padding
        min_pixels=pixels,
        max_pixels=pixels,
    )
    _patch_chat_template(processor)
    return processor


def _patch_chat_template(processor):
    """Sanitize messages on every ``apply_chat_template`` call.

    ``with_transform`` runs before collation, but TRL re-templates generated
    turns as well, so the guard has to live on the processor too.
    """
    orig_apply = processor.apply_chat_template

    def _apply_wrapper(self, messages, *args, **kwargs):
        try:
            messages = sanitize_messages(messages)
        except Exception:
            pass
        return orig_apply(messages, *args, **kwargs)

    processor.apply_chat_template = MethodType(_apply_wrapper, processor)


def device_map_for_ddp():
    """Pin each rank to its own GPU; let HF shard when running single-process."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        return {"": local_rank}
    return "auto"


def load_model(init_checkpoint: str, base_model: str):
    """Load ``init_checkpoint`` if it exists on disk, else the base model."""
    model_path = init_checkpoint if (init_checkpoint and os.path.isdir(init_checkpoint)) else base_model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        device_map=device_map_for_ddp(),
        torch_dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16),
        attn_implementation="flash_attention_2",
    )
    try:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    except Exception:
        pass
    return model
