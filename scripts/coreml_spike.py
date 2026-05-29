#!/usr/bin/env python3
"""Feasibility spike: does CoreML (Apple Neural Engine + GPU) beat eager PyTorch/MPS
for an SD1.5 UNet forward pass?

Builds an SD1.5-shaped UNet, measures eager-MPS fp16 throughput, then converts the
same UNet to a CoreML .mlpackage (fp16) and measures it under different compute units
(ALL = ANE+GPU+CPU, CPU_AND_NE = prefer ANE, CPU_AND_GPU). All at batch=2 (CFG),
512px latents, so "fwd/s" is directly comparable to the app's it/s.

    conda run -n invokeai python scripts/coreml_spike.py
"""

from __future__ import annotations

import time

import numpy as np
import torch
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

LATENT = 64  # 512px / 8
BATCH = 2  # CFG: cond + uncond
ITERS = 20
WARMUP = 3
MLPACKAGE = "/tmp/sd15_unet.mlpackage"


def make_sd15_unet() -> UNet2DConditionModel:
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


class UNetWrapper(torch.nn.Module):
    """Return a plain tensor (CoreML/tracing can't return a dataclass)."""

    def __init__(self, unet: UNet2DConditionModel):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states):
        return self.unet(sample, timestep, encoder_hidden_states).sample


def bench_eager_mps(unet: UNet2DConditionModel) -> float:
    dev = torch.device("mps")
    m = unet.to(device=dev, dtype=torch.float16).eval()
    sample = torch.randn(BATCH, 4, LATENT, LATENT, device=dev, dtype=torch.float16)
    ts = torch.tensor([10] * BATCH, device=dev, dtype=torch.float16)
    ehs = torch.randn(BATCH, 77, 768, device=dev, dtype=torch.float16)
    with torch.inference_mode():
        for _ in range(WARMUP + 2):
            m(sample, ts, encoder_hidden_states=ehs).sample
        torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            m(sample, ts, encoder_hidden_states=ehs).sample
        torch.mps.synchronize()
        dt = time.perf_counter() - t0
    return ITERS / dt


def convert_to_coreml(unet: UNet2DConditionModel) -> None:
    import coremltools as ct

    wrapper = UNetWrapper(unet).to(device="cpu", dtype=torch.float32).eval()
    sample = torch.randn(BATCH, 4, LATENT, LATENT, dtype=torch.float32)
    ts = torch.tensor([10.0] * BATCH, dtype=torch.float32)
    ehs = torch.randn(BATCH, 77, 768, dtype=torch.float32)

    print("Tracing UNet...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (sample, ts, ehs))

    print("Converting to CoreML (fp16)... this can take a few minutes")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="sample", shape=sample.shape),
            ct.TensorType(name="timestep", shape=ts.shape),
            ct.TensorType(name="encoder_hidden_states", shape=ehs.shape),
        ],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS14,
    )
    mlmodel.save(MLPACKAGE)
    print(f"Saved {MLPACKAGE}")


def bench_coreml(compute_unit_name: str) -> float | None:
    import coremltools as ct

    cu = getattr(ct.ComputeUnit, compute_unit_name)
    try:
        m = ct.models.MLModel(MLPACKAGE, compute_units=cu)
    except Exception as e:  # noqa: BLE001
        print(f"  [{compute_unit_name}] load failed: {str(e).splitlines()[-1][:120]}")
        return None
    feed = {
        "sample": np.random.randn(BATCH, 4, LATENT, LATENT).astype(np.float32),
        "timestep": np.array([10.0] * BATCH, dtype=np.float32),
        "encoder_hidden_states": np.random.randn(BATCH, 77, 768).astype(np.float32),
    }
    try:
        for _ in range(WARMUP):
            m.predict(feed)
        t0 = time.perf_counter()
        for _ in range(ITERS):
            m.predict(feed)
        dt = time.perf_counter() - t0
    except Exception as e:  # noqa: BLE001
        print(f"  [{compute_unit_name}] predict failed: {str(e).splitlines()[-1][:120]}")
        return None
    return ITERS / dt


def main() -> None:
    import coremltools as ct

    print(f"torch {torch.__version__}  coremltools {ct.__version__}")
    print(f"SD1.5 UNet, batch={BATCH}, latent {LATENT}x{LATENT} (512px), {ITERS} iters\n")

    unet = make_sd15_unet()

    eager = bench_eager_mps(unet)
    print(f"eager MPS fp16:           {eager:6.2f} fwd/s ({1000 / eager:6.1f} ms/fwd)\n")

    convert_to_coreml(unet)
    print()

    results = {}
    for cu in ("ALL", "CPU_AND_NE", "CPU_AND_GPU"):
        r = bench_coreml(cu)
        results[cu] = r
        if r:
            print(f"CoreML {cu:12s}:    {r:6.2f} fwd/s ({1000 / r:6.1f} ms/fwd)   "
                  f"{'<<< ' + f'{r / eager:.2f}x vs eager' if r else ''}")

    best = max((v for v in results.values() if v), default=None)
    print("\n=== VERDICT ===")
    print(f"eager MPS:  {eager:.2f} fwd/s")
    if best:
        print(f"best CoreML: {best:.2f} fwd/s  =>  {best / eager:.2f}x vs eager MPS")
        print("CoreML is FASTER -- ANE/GPU track is worth pursuing." if best > eager
              else "CoreML is NOT faster than eager MPS here -- track not worth it as-is.")


if __name__ == "__main__":
    main()
