#!/usr/bin/env python3
"""De-risk the DeepCache default across content types (portraits/faces show artifacts most)."""
from __future__ import annotations

import torch
from DeepCache import DeepCacheSDHelper
from diffusers import StableDiffusionXLPipeline

CKPT = "/Users/sev/StableDiffusion/[shared]/Checkpoints/animergePonyXL_v60.safetensors"
PROMPTS = {
    "portrait": "close-up portrait photo of a woman, freckles, detailed eyes, soft light, 85mm",
    "busy": "a bustling medieval marketplace, many people, stalls, intricate detail, wide shot",
}

pipe = StableDiffusionXLPipeline.from_single_file(CKPT, torch_dtype=torch.float16, add_watermarker=False)
pipe.to("mps")
pipe.set_progress_bar_config(disable=True)


def gen(prompt):
    g = torch.Generator("mps").manual_seed(0)
    return pipe(prompt, negative_prompt="blurry, low quality", num_inference_steps=20, guidance_scale=6.0,
                height=1024, width=1024, generator=g).images[0]


gen(PROMPTS["portrait"])  # warmup
for name, prompt in PROMPTS.items():
    gen(prompt).save(f"/tmp/dcq_{name}_base.png")
    h = DeepCacheSDHelper(pipe=pipe)
    h.set_params(cache_interval=3, cache_branch_id=0)
    h.enable()
    gen(prompt).save(f"/tmp/dcq_{name}_dc3.png")
    h.disable()
    print(f"saved {name}: base + dc3")
