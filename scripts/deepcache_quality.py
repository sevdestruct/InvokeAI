#!/usr/bin/env python3
"""Save same-seed SDXL images with/without DeepCache to eyeball the quality cost."""
from __future__ import annotations

import torch
from DeepCache import DeepCacheSDHelper
from diffusers import StableDiffusionXLPipeline

CKPT = "/Users/sev/StableDiffusion/[shared]/Checkpoints/animergePonyXL_v60.safetensors"
PROMPT = "a scenic mountain landscape at sunset, highly detailed, sharp focus"
NEG = "blurry, low quality"
SEED = 0
STEPS = 20

pipe = StableDiffusionXLPipeline.from_single_file(CKPT, torch_dtype=torch.float16, add_watermarker=False)
pipe.to("mps")
pipe.set_progress_bar_config(disable=True)


def gen():
    g = torch.Generator("mps").manual_seed(SEED)
    return pipe(PROMPT, negative_prompt=NEG, num_inference_steps=STEPS, guidance_scale=6.0,
                height=1024, width=1024, generator=g).images[0]


gen()  # warmup
gen().save("/tmp/dc_base.png")
print("saved baseline")
for iv in (2, 3):
    h = DeepCacheSDHelper(pipe=pipe)
    h.set_params(cache_interval=iv, cache_branch_id=0)
    h.enable()
    gen().save(f"/tmp/dc_iv{iv}.png")
    h.disable()
    print(f"saved interval={iv}")
