#!/usr/bin/env python3
"""Hand-rolled ANE-optimized SD1.5 UNet (Apple SPLIT_EINSUM attention) vs eager MPS.

Implements Apple's "Deploying Transformers on the Apple Neural Engine" attention:
channels-first (B, C, 1, S) layout + chunked per-head einsum. This is the surgery
that makes attention ANE-friendly (the naive CoreML conversion fell back off the ANE).
We swap it into a stock SD1.5 UNet, convert to CoreML fp16, and benchmark CPU_AND_NE
(prefer ANE) and ALL against the eager-MPS baseline.

    conda run -n invokeai python scripts/coreml_ane_handroll.py
"""

from __future__ import annotations

import time

import numpy as np
import torch
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

LATENT = 64
BATCH = 2
ITERS = 20
WARMUP = 3
MLPACKAGE = "/tmp/sd15_unet_ane.mlpackage"


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


class SplitEinsumAttnProcessor:
    """diffusers attention processor using Apple's ANE SPLIT_EINSUM formulation."""

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **kwargs):
        residual = hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)  # (B, S, C)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross is not None:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        heads = attn.heads
        dim_head = query.shape[-1] // heads

        # (B, S, inner) -> (B, inner, 1, S)  [channels-first, ANE-friendly]
        def to_bc1s(x):
            return x.transpose(1, 2).unsqueeze(2)

        q = to_bc1s(query)
        k = to_bc1s(key)
        v = to_bc1s(value)

        mh_q = q.split(dim_head, dim=1)
        mh_k = k.split(dim_head, dim=1)
        mh_v = v.split(dim_head, dim=1)
        scale = dim_head**-0.5

        attn_weights = [torch.einsum("bchq,bchk->bkhq", qi, ki) * scale for qi, ki in zip(mh_q, mh_k, strict=False)]
        attn_weights = [w.softmax(dim=1) for w in attn_weights]
        out = [torch.einsum("bkhq,bchk->bchq", wi, vi) for wi, vi in zip(attn_weights, mh_v, strict=False)]
        out = torch.cat(out, dim=1)  # (B, inner, 1, S)
        out = out.squeeze(2).transpose(1, 2)  # (B, S, inner)

        out = attn.to_out[0](out)
        out = attn.to_out[1](out)

        if input_ndim == 4:
            out = out.transpose(1, 2).view(b, c, h, w)
        if attn.residual_connection:
            out = out + residual
        out = out / attn.rescale_output_factor
        return out


class UNetWrapper(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states):
        return self.unet(sample, timestep, encoder_hidden_states).sample


def bench_eager_mps() -> float:
    dev = torch.device("mps")
    m = make_sd15_unet().to(device=dev, dtype=torch.float16).eval()
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


def verify_split_einsum() -> float:
    """Sanity-check the SPLIT_EINSUM processor produces ~same output as default attention."""
    ref = make_sd15_unet().eval()
    sample = torch.randn(BATCH, 4, LATENT, LATENT)
    ts = torch.tensor([10] * BATCH)
    ehs = torch.randn(BATCH, 77, 768)
    with torch.inference_mode():
        out_ref = ref(sample, ts, ehs).sample
        ref.set_attn_processor(SplitEinsumAttnProcessor())
        out_ane = ref(sample, ts, ehs).sample
    return (out_ref - out_ane).abs().max().item()


def convert_to_coreml() -> None:
    import coremltools as ct

    unet = make_sd15_unet().eval()
    unet.set_attn_processor(SplitEinsumAttnProcessor())
    wrapper = UNetWrapper(unet).to(device="cpu", dtype=torch.float32).eval()
    sample = torch.randn(BATCH, 4, LATENT, LATENT, dtype=torch.float32)
    ts = torch.tensor([10.0] * BATCH, dtype=torch.float32)
    ehs = torch.randn(BATCH, 77, 768, dtype=torch.float32)

    print("Tracing ANE-optimized UNet...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (sample, ts, ehs))

    print("Converting to CoreML (fp16)...")
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


def bench_coreml(cu_name: str) -> float | None:
    import coremltools as ct

    try:
        m = ct.models.MLModel(MLPACKAGE, compute_units=getattr(ct.ComputeUnit, cu_name))
    except Exception as e:  # noqa: BLE001
        print(f"  [{cu_name}] load failed: {str(e).splitlines()[-1][:120]}")
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
        print(f"  [{cu_name}] predict failed: {str(e).splitlines()[-1][:120]}")
        return None
    return ITERS / dt


def main() -> None:
    import coremltools as ct

    print(f"torch {torch.__version__}  coremltools {ct.__version__}")
    print(f"Hand-rolled ANE (SPLIT_EINSUM) SD1.5 UNet, batch={BATCH}, 512px, {ITERS} iters\n")

    max_diff = verify_split_einsum()
    print(f"SPLIT_EINSUM vs default attention max abs diff: {max_diff:.4e} (should be tiny)\n")

    eager = bench_eager_mps()
    print(f"eager MPS fp16 (standard attn): {eager:6.2f} fwd/s ({1000 / eager:6.1f} ms/fwd)\n")

    convert_to_coreml()
    print()

    results = {}
    for cu in ("ALL", "CPU_AND_NE"):
        r = bench_coreml(cu)
        results[cu] = r
        if r:
            print(f"CoreML {cu:12s} (ANE-opt): {r:6.2f} fwd/s ({1000 / r:6.1f} ms/fwd)  {r / eager:.2f}x vs eager")

    best = max((v for v in results.values() if v), default=None)
    print("\n=== VERDICT ===")
    print(f"eager MPS (GPU): {eager:.2f} fwd/s")
    if best:
        print(f"best ANE CoreML: {best:.2f} fwd/s  =>  {best / eager:.2f}x vs MPS")
        print("ANE WINS." if best > eager else "MPS still wins on this hardware.")


if __name__ == "__main__":
    main()
