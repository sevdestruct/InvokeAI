#!/usr/bin/env python3
"""Does DeepCache actually speed up SDXL on Apple Silicon (MPS)?

The research says DeepCache skips deep UNet blocks on most steps (real FLOP reduction,
unlike ToMe's token-merge overhead), reusing high-level features and only recomputing
shallow layers. Every published speedup is NVIDIA-measured, so this re-benchmarks it on
THIS M3 Max with end-to-end wall time + output deviation vs the uncached baseline.

    conda run -n invokeai python scripts/deepcache_bench.py
"""

from __future__ import annotations

import statistics
import time

import numpy as np
import torch
from DeepCache import DeepCacheSDHelper
from diffusers import StableDiffusionXLPipeline

CKPT = "/Users/sev/StableDiffusion/[shared]/Checkpoints/animergePonyXL_v60.safetensors"
DEVICE = "mps"
STEPS = 20
H = W = 1024
SEED = 0
PROMPT = "a scenic mountain landscape at sunset, highly detailed, sharp focus"
NEG = "blurry, low quality"


def main() -> None:
    print(f"torch {torch.__version__}; loading SDXL via from_single_file (MPS, fp16)...")
    pipe = StableDiffusionXLPipeline.from_single_file(CKPT, torch_dtype=torch.float16, add_watermarker=False)
    pipe.to(DEVICE)
    pipe.set_progress_bar_config(disable=True)

    def gen():
        g = torch.Generator(DEVICE).manual_seed(SEED)
        return pipe(
            PROMPT, negative_prompt=NEG, num_inference_steps=STEPS, guidance_scale=6.0,
            height=H, width=W, generator=g,
        ).images[0]

    def timeit(n=2):
        ts, img = [], None
        for _ in range(n):
            t0 = time.time()
            img = gen()
            ts.append(time.time() - t0)
        return statistics.median(ts), img

    print("warmup...")
    gen()
    base_t, base_img = timeit()
    base = np.asarray(base_img).astype(np.float32)
    print(f"\nbaseline {STEPS}-step SDXL {W}px (no cache): {base_t:.2f}s\n")

    for interval in (2, 3, 5):
        helper = DeepCacheSDHelper(pipe=pipe)
        helper.set_params(cache_interval=interval, cache_branch_id=0)
        helper.enable()
        t, img = timeit()
        helper.disable()
        arr = np.asarray(img).astype(np.float32)
        dev = np.abs(arr - base).mean() / (base.mean() + 1e-6) * 100.0
        print(f"DeepCache interval={interval}: {t:6.2f}s  {base_t / t:.2f}x faster  | mean pixel deviation {dev:4.1f}%")

    print("\n(interval=N: deep UNet blocks recomputed every Nth step, reused otherwise)")


if __name__ == "__main__":
    main()
