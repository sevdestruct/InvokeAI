#!/usr/bin/env python3
"""Micro-benchmark of an SD1.5-shaped UNet forward pass on MPS.

Isolates raw UNet compute (the dominant cost of denoising) from all of InvokeAI's
machinery, so we can measure the true MPS ceiling and the effect of individual
optimizations (channels_last, dtype, torch.compile) in seconds rather than via
full generations.

The denoise loop does one batch=2 forward per step (CFG), so "batched fwd/s" here
is directly comparable to the app's reported it/s.

    conda run -n invokeai python scripts/micro_bench_unet.py
"""

from __future__ import annotations

import time

import torch
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

DEVICE = torch.device("mps")
LATENT = 64  # 512px / 8
ITERS = 20
WARMUP = 5


def make_sd15_unet() -> UNet2DConditionModel:
    # Standard SD1.5 UNet config. Random weights -- timing is weight-independent.
    return UNet2DConditionModel(
        sample_size=LATENT,
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


def bench(label: str, unet, dtype, channels_last: bool, compile_kwargs: dict | None) -> None:
    unet = unet.to(device=DEVICE, dtype=dtype)
    unet.eval()
    if channels_last:
        unet = unet.to(memory_format=torch.channels_last)

    fwd = torch.compile(unet, **compile_kwargs) if compile_kwargs else unet

    batch = 2  # CFG: cond + uncond in one forward
    sample = torch.randn(batch, 4, LATENT, LATENT, device=DEVICE, dtype=dtype)
    if channels_last:
        sample = sample.to(memory_format=torch.channels_last)
    timestep = torch.tensor([10] * batch, device=DEVICE, dtype=torch.float16 if dtype == torch.float16 else dtype)
    ehs = torch.randn(batch, 77, 768, device=DEVICE, dtype=dtype)

    try:
        with torch.inference_mode():
            for _ in range(WARMUP):
                fwd(sample, timestep, encoder_hidden_states=ehs).sample
            torch.mps.synchronize()
            t0 = time.perf_counter()
            for _ in range(ITERS):
                fwd(sample, timestep, encoder_hidden_states=ehs).sample
            torch.mps.synchronize()
            dt = time.perf_counter() - t0
    except Exception as e:  # noqa: BLE001
        msg = str(e).splitlines()[-1][:90] if str(e) else type(e).__name__
        print(f"{label:42s}  FAILED: {msg}")
        return

    fwd_per_s = ITERS / dt
    print(f"{label:42s}  {fwd_per_s:6.2f} batched-fwd/s  ({dt / ITERS * 1000:6.1f} ms/fwd)")


def main() -> None:
    print(f"torch {torch.__version__}  mps={torch.backends.mps.is_available()}  device={DEVICE}")
    print(f"SD1.5 UNet, latent {LATENT}x{LATENT} (512px), batch=2, {ITERS} iters after {WARMUP} warmup\n")

    unet = make_sd15_unet()
    bench("eager fp16", unet, torch.float16, channels_last=False, compile_kwargs=None)
    bench("eager fp16 + channels_last", unet, torch.float16, channels_last=True, compile_kwargs=None)
    bench("eager bf16", unet, torch.bfloat16, channels_last=False, compile_kwargs=None)
    bench("eager fp32", unet, torch.float32, channels_last=False, compile_kwargs=None)
    bench("compile fp16 aot_eager", unet, torch.float16, channels_last=False,
          compile_kwargs={"backend": "aot_eager", "fullgraph": False, "dynamic": False})
    bench("compile fp16 inductor", unet, torch.float16, channels_last=False,
          compile_kwargs={"mode": "default", "fullgraph": False, "dynamic": False})


if __name__ == "__main__":
    main()
