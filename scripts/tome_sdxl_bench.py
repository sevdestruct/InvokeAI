#!/usr/bin/env python3
"""VALID ToMe re-test: token merging on a real SDXL diffusers pipeline on MPS.

The earlier in-app ToMe result was invalid (ToMe was wired into the modular denoise
path but the default path ran, so it never applied). This applies tomesd directly to a
real SDXL pipeline's UNet and measures end-to-end speed + same-seed quality at a few
merge ratios -- the honest test ToMe deserved.

    conda run -n invokeai python scripts/tome_sdxl_bench.py
"""

from __future__ import annotations

import statistics
import time

import numpy as np
import tomesd
import torch
from diffusers import StableDiffusionXLPipeline

CKPT = "/Users/sev/StableDiffusion/[shared]/Checkpoints/animergePonyXL_v60.safetensors"
PROMPT = "a scenic mountain landscape at sunset, highly detailed, sharp focus"
NEG = "blurry, low quality"


def main() -> None:
    print(f"torch {torch.__version__}; loading SDXL (MPS, fp16)...")
    pipe = StableDiffusionXLPipeline.from_single_file(CKPT, torch_dtype=torch.float16, add_watermarker=False)
    pipe.to("mps")
    pipe.set_progress_bar_config(disable=True)

    def gen():
        g = torch.Generator("mps").manual_seed(0)
        return pipe(PROMPT, negative_prompt=NEG, num_inference_steps=20, guidance_scale=6.0,
                    height=1024, width=1024, generator=g).images[0]

    def timeit(n=2):
        ts, img = [], None
        for _ in range(n):
            t0 = time.time()
            img = gen()
            ts.append(time.time() - t0)
        return statistics.median(ts), img

    print("warmup..."); gen()
    base_t, base_img = timeit()
    base = np.asarray(base_img).astype(np.float32)
    print(f"\nbaseline SDXL 1024 20-step (no ToMe): {base_t:.2f}s\n")

    for ratio in (0.25, 0.5, 0.75):
        tomesd.apply_patch(pipe, ratio=ratio)
        t, img = timeit()
        tomesd.remove_patch(pipe)
        arr = np.asarray(img).astype(np.float32)
        dev = np.abs(arr - base).mean() / (base.mean() + 1e-6) * 100.0
        print(f"ToMe ratio {ratio:.2f}: {t:6.2f}s  {base_t / t:.2f}x  | pixel dev {dev:4.1f}%")

    print("\n(real SDXL pipeline; ToMe applied via tomesd.apply_patch on the UNet)")


if __name__ == "__main__":
    main()
