#!/usr/bin/env python3
"""Save the maximal-stack image (Hyper + DeepCache + TAESD) to verify quality."""
from __future__ import annotations

import torch
from DeepCache import DeepCacheSDHelper
from diffusers import AutoencoderTiny, StableDiffusionPipeline

CKPT = "/Users/sev/StableDiffusion/[shared]/Checkpoints/realisticVisionV60B1_v51HyperVAE.safetensors"
pipe = StableDiffusionPipeline.from_single_file(CKPT, torch_dtype=torch.float16, safety_checker=None)
pipe.to("mps")
pipe.set_progress_bar_config(disable=True)
pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float16).to("mps")
h = DeepCacheSDHelper(pipe=pipe)
h.set_params(cache_interval=2, cache_branch_id=0)
h.enable()
g = torch.Generator("mps").manual_seed(0)
img = pipe("a scenic mountain landscape at sunset, highly detailed, sharp focus",
           negative_prompt="blurry, low quality", num_inference_steps=6, guidance_scale=1.5,
           height=512, width=512, generator=g).images[0]
img.save("/tmp/maxstack.png")
print("saved /tmp/maxstack.png (Hyper 6-step + DeepCache(2) + TAESD)")
