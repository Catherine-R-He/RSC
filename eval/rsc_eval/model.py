"""Qwen2.5-VL inference wrapper for RSC evaluation."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoProcessor,
    GenerationConfig,
    Qwen2_5_VLForConditionalGeneration,
)

from .parsing import parse_model_output

_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}

PROMPT_TEMPLATE = (
    "<image>\n"
    "You are given an image and a referring scenario description.\n"
    "First, think step by step inside <think>...</think>.\n"
    "Then output ONE JSON object inside <answer>...</answer>:\n"
    "<answer>{{\n"
    '  "target_object": "<canonical {vocab} category>",\n'
    '  "bbox": [x, y, w, h]\n'
    "}}</answer>\n"
    "Rules:\n"
    "- The only content outside <answer> should be <think>…</think>.\n"
    '- "bbox" are 4 INTEGERS [x,y,w,h], xywh in pixels, clipped to bounds.\n'
    "Image size: Height={H}, Width={W}\n"
    "Referring scenario description:\n"
    "{scenario}\n"
)


@dataclass
class ModelCfg:
    model_path: str
    processor_path: Optional[str] = None       # defaults to base_model_id
    # A GRPO/DeepSpeed checkpoint dir can carry a config.json that makes
    # transformers take the seq2seq path, so the architecture config and the
    # generation config are read from the base model and only the weights come
    # from model_path.
    base_model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    dtype: str = "bfloat16"
    device: str = "cuda:0"
    max_new_tokens: int = 512
    # Left as-is from the released runs: Qwen2.5-VL's own generation_config
    # enables sampling, and these values make it effectively greedy. Changing
    # them changes the reported numbers.
    temperature: float = 0.01
    top_p: float = 1.0
    attn_impl: str = "flash_attention_2"       # or "eager"
    vocab_label: str = "COCO"                  # "COCO" (in-domain) or "LVIS" (OOD)


class QwenVL:
    def __init__(self, cfg: ModelCfg):
        self.cfg = cfg
        dt = _DTYPES[cfg.dtype]
        base_src = cfg.base_model_id
        proc_src = cfg.processor_path or base_src

        self.processor = AutoProcessor.from_pretrained(proc_src, trust_remote_code=True)
        clean_cfg = AutoConfig.from_pretrained(base_src, trust_remote_code=True)
        gen_cfg = GenerationConfig.from_pretrained(base_src)

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            cfg.model_path,
            config=clean_cfg,
            generation_config=gen_cfg,
            torch_dtype=dt,
            device_map=cfg.device,
            trust_remote_code=True,
            attn_implementation=cfg.attn_impl,
        )
        self.model.eval()

    def predict_box(self, image: Image.Image, scenario: str, H: int, W: int,
                    expect_norm: bool = False
                    ) -> Tuple[Optional[str], Optional[str], Optional[List[int]]]:
        prompt = PROMPT_TEMPLATE.format(vocab=self.cfg.vocab_label, H=H, W=W, scenario=scenario)
        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}],
        }]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
            )

        decoded = self.processor.batch_decode(out_ids, skip_special_tokens=True)[0]
        return parse_model_output(decoded, H=H, W=W, expect_norm=expect_norm)
