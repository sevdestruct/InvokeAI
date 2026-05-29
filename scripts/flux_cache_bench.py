#!/usr/bin/env python3
"""Does FLUX DiT caching (FirstBlockCache) speed up FLUX on MPS? — speed + quality.

FLUX is a DiT (transformer), so the UNet-oriented DeepCache doesn't apply; diffusers
ships transformer caching (FirstBlockCache: skip later blocks when the first block's
output barely changes). Caching pays off at higher step counts, so we run FLUX.1-schnell
at a configurable step count and compare baseline vs FBCache at a couple thresholds.

    conda run -n invokeai python scripts/flux_cache_bench.py --steps 28
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np
import torch
from diffusers import FirstBlockCacheConfig, FluxPipeline

PROMPT = "a scenic mountain landscape at sunset, highly detailed, sharp focus"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--size", type=int, default=1024)
    args = ap.parse_args()

    print(f"torch {torch.__version__}; loading FLUX.1-schnell (MPS, bf16)...")
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
    pipe.to("mps")
    pipe.set_progress_bar_config(disable=True)

    def gen():
        g = torch.Generator("mps").manual_seed(0)
        return pipe(PROMPT, num_inference_steps=args.steps, guidance_scale=0.0,
                    height=args.size, width=args.size, generator=g).images[0]

    def timeit(n=1):  # FLUX is slow; one timed run after warmup is enough for the ratio
        ts, img = [], None
        for _ in range(n):
            t0 = time.time()
            img = gen()
            ts.append(time.time() - t0)
        return statistics.median(ts), img

    print("warmup..."); gen()
    base_t, base_img = timeit()
    base_img.save("/tmp/flux_base.png")
    base = np.asarray(base_img).astype(np.float32)
    print(f"\nbaseline FLUX schnell {args.size}px {args.steps}-step: {base_t:.2f}s\n")

    for thr in (0.12, 0.25):
        pipe.transformer.enable_cache(FirstBlockCacheConfig(threshold=thr))
        t, img = timeit()
        pipe.transformer.disable_cache()
        arr = np.asarray(img).astype(np.float32)
        dev = np.abs(arr - base).mean() / (base.mean() + 1e-6) * 100.0
        img.save(f"/tmp/flux_fbc_{thr}.png")
        print(f"FirstBlockCache thr={thr}: {t:6.2f}s  {base_t / t:.2f}x  | pixel dev {dev:4.1f}%")

    print("\nimages: /tmp/flux_base.png vs /tmp/flux_fbc_*.png")


if __name__ == "__main__":
    main()
