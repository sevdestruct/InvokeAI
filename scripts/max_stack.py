#!/usr/bin/env python3
"""Maximal achievable stack on SD1.5: few-step (Hyper) + DeepCache + TAESD, on MPS."""
from __future__ import annotations

import statistics
import time

import torch
from DeepCache import DeepCacheSDHelper
from diffusers import AutoencoderTiny, StableDiffusionPipeline

CKPT = "/Users/sev/StableDiffusion/[shared]/Checkpoints/realisticVisionV60B1_v51HyperVAE.safetensors"
PROMPT = "a scenic mountain landscape at sunset, highly detailed, sharp focus"
NEG = "blurry, low quality"
STEPS = 6  # Hyper model


def main() -> None:
    pipe = StableDiffusionPipeline.from_single_file(CKPT, torch_dtype=torch.float16, safety_checker=None)
    pipe.to("mps")
    pipe.set_progress_bar_config(disable=True)
    full_vae = pipe.vae
    taesd = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float16).to("mps")

    def gen():
        g = torch.Generator("mps").manual_seed(0)
        return pipe(PROMPT, negative_prompt=NEG, num_inference_steps=STEPS, guidance_scale=1.5,
                    height=512, width=512, generator=g).images[0]

    def timeit(n=3):
        ts = []
        for _ in range(n):
            t0 = time.time()
            gen()
            ts.append(time.time() - t0)
        return statistics.median(ts)

    print("warmup..."); gen()
    rows = []
    pipe.vae = full_vae
    rows.append(("Hyper 6-step  full-VAE  (already ~3x vs std-20-step)", timeit()))

    pipe.vae = taesd
    rows.append(("Hyper 6-step  + TAESD", timeit()))

    h = DeepCacheSDHelper(pipe=pipe); h.set_params(cache_interval=2, cache_branch_id=0); h.enable()
    rows.append(("Hyper 6-step  + TAESD + DeepCache(2)", timeit()))
    h.disable()

    print("\n=== Maximal stack (SD1.5 Hyper, 512px) ===")
    base = rows[0][1]
    for name, t in rows:
        ips = STEPS / t  # rough images/sec proxy via total time
        print(f"  {name:52s} {t:5.2f}s/img  ({1 / t:4.2f} img/s)  {base / t:.2f}x")
    print(f"\n(reference: standard SD1.5 20-step full-VAE was ~4.7s/img)")


if __name__ == "__main__":
    main()
