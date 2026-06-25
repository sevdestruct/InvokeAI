#!/usr/bin/env python3
"""On-device roofline check: is FLUX DiT denoise compute-bound or bandwidth-bound on THIS Mac?

The research concluded compute-bound (FLUX GEMMs ~5 orders of magnitude above the M3 Max roofline
ridge), which determines whether weight quantization can ever be a speed lever (it cannot, if
compute-bound). This settles it empirically on the user's chip: measure achieved bf16 TFLOPS for
FLUX-representative FFN/QKV GEMMs and compare to the peak square-matmul TFLOPS. High utilization =>
compute-bound => quantization is footprint-only.

  conda run -n invokeai python scripts/roofline_check.py
"""

from __future__ import annotations

import time

import torch

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
DT = torch.bfloat16


def tflops(m: int, k: int, n: int, iters: int = 30) -> float:
    a = torch.randn(m, k, device=DEV, dtype=DT)
    b = torch.randn(k, n, device=DEV, dtype=DT)
    for _ in range(3):  # warmup
        (a @ b)
    torch.mps.synchronize() if DEV == "mps" else None
    t0 = time.time()
    acc = None
    for _ in range(iters):
        acc = a @ b
    if acc is not None and DEV == "mps":
        torch.mps.synchronize()
    dt = time.time() - t0
    return (iters * 2 * m * k * n) / dt / 1e12


def bandwidth_gbps(n: int = 64_000_000, iters: int = 50) -> float:
    # Memory-bound elementwise: reads + writes dominate (low arithmetic intensity).
    x = torch.randn(n, device=DEV, dtype=DT)
    for _ in range(3):
        x = x * 1.0001 + 0.0
    torch.mps.synchronize() if DEV == "mps" else None
    t0 = time.time()
    for _ in range(iters):
        x = x * 1.0001 + 0.0
    torch.mps.synchronize() if DEV == "mps" else None
    dt = time.time() - t0
    bytes_moved = iters * n * 2 * 2  # read+write, 2 bytes/elem
    return bytes_moved / dt / 1e9


def main() -> None:
    print(f"device={DEV} dtype={DT}\n")
    peak = tflops(8192, 8192, 8192)
    print(f"peak square matmul (8192^3)              : {peak:6.1f} TFLOPS  (proxy for compute ceiling)")

    # FLUX.1 hidden=3072. Representative per-step GEMMs at seq ~4352 (1024px packed img + 256 txt).
    seq = 4352
    flux_gemms = {
        "FLUX qkv     [seq,3072]@[3072,9216]": (seq, 3072, 9216),
        "FLUX ffn-in  [seq,3072]@[3072,12288]": (seq, 3072, 12288),
        "FLUX ffn-out [seq,12288]@[12288,3072]": (seq, 12288, 3072),
        "FLUX proj    [seq,3072]@[3072,3072]": (seq, 3072, 3072),
    }
    print()
    achieved = []
    for name, (m, k, n) in flux_gemms.items():
        tf = tflops(m, k, n)
        achieved.append(tf)
        print(f"  {name:42} {tf:6.1f} TFLOPS  ({tf / peak * 100:4.0f}% of peak)")

    bw = bandwidth_gbps()
    print(f"\nmemory-bound elementwise bandwidth       : {bw:6.0f} GB/s  (M3 Max spec ~400 GB/s)")

    util = sum(achieved) / len(achieved) / peak
    print("\n=== verdict ===")
    print(f"mean FLUX-GEMM utilization of compute peak: {util * 100:.0f}%")
    if util > 0.55:
        print("COMPUTE-BOUND confirmed: GEMMs run near the compute ceiling, so the denoise step is")
        print("limited by FLOPs, not memory traffic. Weight quantization shrinks bytes that are NOT")
        print("the bottleneck -> footprint-only, no latency win. Levers: fewer steps (caching) +")
        print("faster FP attention kernels (flash-attention).")
    else:
        print("Not clearly compute-bound on these shapes -> a quantization spike may be warranted.")


if __name__ == "__main__":
    main()
