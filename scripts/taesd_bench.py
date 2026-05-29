#!/usr/bin/env python3
"""TAESD (Tiny AutoEncoder) vs the full SDXL VAE decode on MPS — speed + quality.

The VAE decode is a fixed per-image cost (~1s on SDXL), so once denoise is fast
(DeepCache / few-step models) it becomes a large fraction of total time. TAESD is a
tiny distilled autoencoder that decodes the same latents in a fraction of the time.
We isolate VAE decode time (total - latent-only) for the full VAE vs TAESD, and save
both images to eyeball quality.

    conda run -n invokeai python scripts/taesd_bench.py
"""

from __future__ import annotations

import statistics
import time

import torch
from diffusers import AutoencoderTiny, StableDiffusionXLPipeline

CKPT = "/Users/sev/StableDiffusion/[shared]/Checkpoints/animergePonyXL_v60.safetensors"
PROMPT = "a scenic mountain landscape at sunset, highly detailed, sharp focus"
NEG = "blurry, low quality"
SEED = 0
STEPS = 8


def main() -> None:
    print(f"torch {torch.__version__}; loading SDXL (MPS, fp16)...")
    pipe = StableDiffusionXLPipeline.from_single_file(CKPT, torch_dtype=torch.float16, add_watermarker=False)
    pipe.to("mps")
    pipe.set_progress_bar_config(disable=True)

    def run(output_type):
        g = torch.Generator("mps").manual_seed(SEED)
        return pipe(PROMPT, negative_prompt=NEG, num_inference_steps=STEPS, guidance_scale=6.0,
                    height=1024, width=1024, generator=g, output_type=output_type)

    def timeit(output_type, n=3):
        ts, out = [], None
        for _ in range(n):
            t0 = time.time()
            out = run(output_type)
            ts.append(time.time() - t0)
        return statistics.median(ts), out

    print("warmup..."); run("latent")

    t_latent, _ = timeit("latent")            # denoise only (no VAE)
    t_full, out_full = timeit("pil")          # denoise + full VAE
    out_full.images[0].save("/tmp/taesd_full.png")
    vae_full = t_full - t_latent

    # swap in TAESD and re-decode
    taesd = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch.float16).to("mps")
    pipe.vae = taesd
    t_taesd, out_taesd = timeit("pil")
    out_taesd.images[0].save("/tmp/taesd_tiny.png")
    vae_taesd = t_taesd - t_latent

    print(f"\ndenoise-only ({STEPS} steps):     {t_latent:6.2f}s")
    print(f"full SDXL VAE decode:        {vae_full:6.3f}s   (total {t_full:.2f}s)")
    print(f"TAESD decode:                {vae_taesd:6.3f}s   (total {t_taesd:.2f}s)")
    if vae_taesd > 0:
        print(f"VAE decode speedup: {vae_full / max(vae_taesd, 1e-3):.1f}x  (saves ~{vae_full - vae_taesd:.2f}s/image)")
    print("images: /tmp/taesd_full.png vs /tmp/taesd_tiny.png")


if __name__ == "__main__":
    main()
