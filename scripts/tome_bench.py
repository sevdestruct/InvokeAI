#!/usr/bin/env python3
"""Token Merging (ToMe) speedup vs quality on the SD1.5 UNet, on MPS.

ToMe (tomesd) merges redundant spatial tokens before attention, cutting attention
compute. It's model-agnostic (no retraining, works on existing checkpoints) and the
benefit grows with token count (higher resolution). We measure forward throughput and
the output deviation from the unmerged baseline (the quality cost) at a few ratios.

    conda run -n invokeai python scripts/tome_bench.py
"""

from __future__ import annotations

import time

import tomesd
import torch
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

DEVICE = torch.device("mps")
BATCH = 2
ITERS = 20
WARMUP = 5


def make_sd15_unet():
    return UNet2DConditionModel(
        sample_size=64,
        in_channels=4,
        out_channels=4,
        layers_per_block=2,
        block_out_channels=(320, 640, 1280, 1280),
        down_block_types=(
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=768,
        attention_head_dim=8,
    )


def timed(unet, sample, ts, ehs):
    with torch.inference_mode():
        for _ in range(WARMUP):
            unet(sample, ts, encoder_hidden_states=ehs).sample
        torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            out = unet(sample, ts, encoder_hidden_states=ehs).sample
        torch.mps.synchronize()
        dt = time.perf_counter() - t0
    return ITERS / dt, out


def run_res(unet, latent: int, px: int):
    sample = torch.randn(BATCH, 4, latent, latent, device=DEVICE, dtype=torch.float16)
    ts = torch.tensor([10] * BATCH, device=DEVICE, dtype=torch.float16)
    ehs = torch.randn(BATCH, 77, 768, device=DEVICE, dtype=torch.float16)

    base_fwd, base_out = timed(unet, sample, ts, ehs)
    print(f"\n{px}px (latent {latent}², {latent * latent} tokens):")
    print(f"  baseline (no ToMe):  {base_fwd:6.2f} fwd/s ({1000 / base_fwd:6.1f} ms)")

    for ratio in (0.25, 0.5, 0.75):
        tomesd.apply_patch(unet, ratio=ratio)
        fwd, out = timed(unet, sample, ts, ehs)
        diff = (out.float() - base_out.float()).abs().mean().item()
        rel = diff / base_out.float().abs().mean().item()
        tomesd.remove_patch(unet)
        print(f"  ToMe ratio {ratio:.2f}:      {fwd:6.2f} fwd/s ({1000 / fwd:6.1f} ms)  "
              f"{fwd / base_fwd:.2f}x  | mean output dev: {rel * 100:5.1f}%")


def main():
    print(f"torch {torch.__version__}  ToMe (tomesd) on MPS, SD1.5 UNet, batch={BATCH}")
    unet = make_sd15_unet().to(device=DEVICE, dtype=torch.float16).eval()
    run_res(unet, 64, 512)
    run_res(unet, 96, 768)
    run_res(unet, 128, 1024)


if __name__ == "__main__":
    main()
