#!/usr/bin/env python3
"""Tune DeepCache and measure it stacked with TAESD (the combined win)."""
from __future__ import annotations

import statistics
import time

import torch
from DeepCache import DeepCacheSDHelper
from diffusers import AutoencoderTiny, StableDiffusionXLPipeline

CKPT = "/Users/sev/StableDiffusion/[shared]/Checkpoints/animergePonyXL_v60.safetensors"
PROMPT = "a scenic mountain landscape at sunset, highly detailed, sharp focus"
NEG = "blurry, low quality"


def main() -> None:
    pipe = StableDiffusionXLPipeline.from_single_file(CKPT, torch_dtype=torch.float16, add_watermarker=False)
    pipe.to("mps")
    pipe.set_progress_bar_config(disable=True)
    full_vae = pipe.vae
    taesd = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch.float16).to("mps")

    def gen(steps):
        g = torch.Generator("mps").manual_seed(0)
        return pipe(PROMPT, negative_prompt=NEG, num_inference_steps=steps, guidance_scale=6.0,
                    height=1024, width=1024, generator=g).images[0]

    def timeit(steps, n=2):
        ts, img = [], None
        for _ in range(n):
            t0 = time.time()
            img = gen(steps)
            ts.append(time.time() - t0)
        return statistics.median(ts), img

    def dc(interval):
        h = DeepCacheSDHelper(pipe=pipe)
        h.set_params(cache_interval=interval, cache_branch_id=0)
        h.enable()
        return h

    print("warmup..."); gen(8)
    rows = []

    pipe.vae = full_vae
    t, _ = timeit(20); rows.append(("baseline  20-step  full-VAE", t))

    h = dc(3); t, _ = timeit(20); h.disable(); rows.append(("DeepCache=3  20-step  full-VAE", t))

    pipe.vae = taesd
    h = dc(3); t, img = timeit(20); h.disable(); img.save("/tmp/dctaesd_20.png")
    rows.append(("DeepCache=3  20-step  TAESD  (STACKED)", t))

    h = dc(3); t, img = timeit(30); h.disable(); img.save("/tmp/dctaesd_30.png")
    rows.append(("DeepCache=3  30-step  TAESD", t))

    base = rows[0][1]
    print("\n=== SDXL 1024 stacked tuning ===")
    for name, t in rows:
        print(f"  {name:42s} {t:6.2f}s   {base / t:.2f}x")
    print("images: /tmp/dctaesd_20.png  /tmp/dctaesd_30.png")


if __name__ == "__main__":
    main()
