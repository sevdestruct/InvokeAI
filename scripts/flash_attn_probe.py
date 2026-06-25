#!/usr/bin/env python3
"""Probe candidate Metal flash-attention kernels for FLUX on Apple Silicon -- the SAFE, isolated way
to answer "does a fused kernel actually beat PyTorch-MPS SDPA on this chip, and is it numerically
correct?" BEFORE wiring anything into InvokeAI's denoise path or installing into the working env.

It does NOT install anything. It checks which candidate kernels are importable in the CURRENT
environment, then for each: verifies numerical parity vs torch SDPA on FLUX-shaped Q/K/V, and
benchmarks it against SDPA. Run it in an isolated env where you've installed a candidate kernel:

    conda run -n <isolated-env> python scripts/flash_attn_probe.py

If a kernel is both correct (parity OK) and faster, add an adapter for it in
invokeai/backend/flux/math.py:_load_fused_attn_kernel() and enable flux_flash_attention.
"""

from __future__ import annotations

import importlib
import time

import torch

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
DT = torch.bfloat16 if DEV == "mps" else torch.float32

# FLUX schnell @ 1024px: 24 heads, head_dim 128, ~4352 tokens (img+txt). Single CFG stream (B=1).
SHAPES = [(1, 24, 4352, 128), (1, 24, 1024, 128)]

# (module, attribute, adapter) candidates. Adapter maps (q,k,v in [B,H,L,D]) -> out [B,H,L,D].
# Add rows here as you find kernels to try; unavailable ones are skipped, not errors.
CANDIDATES = [
    # ("some_metal_flash_attn", "flash_attn", lambda fn, q, k, v: fn(q, k, v)),
]


def sdpa(q, k, v):
    return torch.nn.functional.scaled_dot_product_attention(q, k, v)


def bench(fn, q, k, v, iters=20):
    for _ in range(3):
        fn(q, k, v)
    if DEV == "mps":
        torch.mps.synchronize()
    t0 = time.time()
    for _ in range(iters):
        out = fn(q, k, v)
    if DEV == "mps":
        torch.mps.synchronize()
    return (time.time() - t0) / iters * 1000.0, out  # ms/call


def main() -> None:
    print(f"device={DEV} dtype={DT}\n")
    available = []
    for mod, attr, adapter in CANDIDATES:
        try:
            m = importlib.import_module(mod)
            fn = getattr(m, attr)
            available.append((f"{mod}.{attr}", lambda q, k, v, fn=fn, a=adapter: a(fn, q, k, v)))
        except Exception as e:
            print(f"skip {mod}.{attr}: not importable ({type(e).__name__})")
    if not available:
        print("\nNo candidate kernels importable in this env. Install one in an ISOLATED env and add it")
        print("to CANDIDATES, then re-run. (Never pip-install experimental kernels into the working")
        print("invokeai env -- it has broken before.)")
        print("Leads to try: kernels-community/metal-flash-sdpa, mps-flash-attn, attention-mps-torch.")
        return

    for shape in SHAPES:
        g = torch.Generator().manual_seed(0)
        q = torch.randn(shape, generator=g).to(DEV, DT)
        k = torch.randn(shape, generator=g).to(DEV, DT)
        v = torch.randn(shape, generator=g).to(DEV, DT)
        base_ms, ref = bench(sdpa, q, k, v)
        print(f"\nshape {shape}:  SDPA baseline {base_ms:.2f} ms/call")
        for name, fn in available:
            try:
                got = fn(q, k, v)
                parity = torch.allclose(got.float(), ref.float(), atol=2e-2, rtol=2e-2)
                ms, _ = bench(fn, q, k, v)
                tag = "PARITY-FAIL" if not parity else f"{base_ms / ms:.2f}x"
                print(f"  {name:32} {ms:7.2f} ms/call  {tag}")
            except Exception as e:
                print(f"  {name:32} ERROR {type(e).__name__}: {str(e)[:50]}")


if __name__ == "__main__":
    main()
