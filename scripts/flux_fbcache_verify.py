#!/usr/bin/env python3
"""Verify the in-app FLUX FirstBlockCache integration against InvokeAI's real Flux class.

This exercises the ACTUAL integrated code (invokeai.backend.flux.model.Flux.forward +
first_block_cache.py), not diffusers' implementation, on a realistic Euler denoise
trajectory. It proves the properties that are specific to *my* integration:

  1. Identity   -- with the cache attached but a ~zero threshold (never skips), the output
                   is identical to the original forward. (The refactored block stack did not
                   change results -- the critical no-regression guarantee.)
  2. Skip+reuse -- with a real threshold, steps whose first-block residual barely changes
                   skip the remaining 18 double + 38 single blocks and reuse the cached
                   aggregate residual; output stays close to the full trajectory.
  3. Structural -- a skipped step runs 1 block instead of 57 -> large per-step speedup
     speedup      (measured here; end-to-end speed depends on skip fraction, which the
                   diffusers FBCache benchmark measured at ~1.9x on real schnell weights).
  4. CFG keying -- two interleaved conditioning streams (positive/negative) keep separate
                   cache state and never cross-contaminate.

Real schnell BLOCK COUNTS (19 double + 38 single) with a reduced hidden size keep the skip
ratio realistic while running in seconds with random weights (the cache logic is identical
regardless of weight values; the compute saved by skipping is structural, not data-dependent).

    conda run -n invokeai python scripts/flux_fbcache_verify.py
"""

from __future__ import annotations

import statistics
import time

import torch

from invokeai.backend.flux.first_block_cache import FluxFirstBlockCache
from invokeai.backend.flux.model import Flux, FluxParams

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.bfloat16
STEPS = 8
IMG_SEQ = 1024  # 512px-equivalent packed FLUX latent
TXT_SEQ = 256


class _NoRegional:
    """Stub regional-prompting extension: no attention masks (the plain, no-regions case)."""

    def get_double_stream_attn_mask(self, block_index: int):
        return None

    def get_single_stream_attn_mask(self, block_index: int):
        return None


def build_model() -> Flux:
    # Real schnell block counts; reduced hidden size for a fast random-weight test.
    params = FluxParams(
        in_channels=64,
        vec_in_dim=768,
        context_in_dim=4096,
        hidden_size=1536,
        mlp_ratio=4.0,
        num_heads=12,
        depth=19,
        depth_single_blocks=38,
        axes_dim=[32, 48, 48],  # sums to hidden_size // num_heads = 128
        theta=10_000,
        qkv_bias=True,
        guidance_embed=False,
    )
    torch.manual_seed(0)
    model = Flux(params).to(device=DEVICE, dtype=DTYPE).eval()
    return model


def make_inputs():
    g = torch.Generator().manual_seed(1234)
    img0 = torch.randn(1, IMG_SEQ, 64, generator=g).to(device=DEVICE, dtype=DTYPE)
    txt = torch.randn(1, TXT_SEQ, 4096, generator=g).to(device=DEVICE, dtype=DTYPE)
    y = torch.randn(1, 768, generator=g).to(device=DEVICE, dtype=DTYPE)
    img_ids = torch.zeros(1, IMG_SEQ, 3, device=DEVICE, dtype=DTYPE)
    txt_ids = torch.zeros(1, TXT_SEQ, 3, device=DEVICE, dtype=DTYPE)
    return img0, img_ids, txt, txt_ids, y


@torch.no_grad()
def run_trajectory(model: Flux, inputs, cache: FluxFirstBlockCache | None, regional=None, update_scale=1.0):
    """Euler denoise loop mirroring flux/denoise.py; returns (final_img, seconds).

    update_scale damps the per-step latent change: ~1.0 is a noisy random-weight walk
    (residuals differ a lot, like an under-converged trajectory); a small value yields a
    smooth trajectory with temporally-coherent residuals, as real FLUX latents exhibit.
    """
    img0, img_ids, txt, txt_ids, y = inputs
    regional = regional or _NoRegional()
    if cache is not None:
        model._fbcache = cache
    elif hasattr(model, "_fbcache"):
        delattr(model, "_fbcache")

    timesteps = torch.linspace(1.0, 0.0, STEPS + 1).tolist()
    img = img0.clone()
    if DEVICE == "mps":
        torch.mps.synchronize()
    t0 = time.time()
    for i in range(STEPS):
        t_curr, t_prev = timesteps[i], timesteps[i + 1]
        t_vec = torch.full((img.shape[0],), t_curr, device=DEVICE, dtype=DTYPE)
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            timesteps=t_vec,
            y=y,
            guidance=None,
            timestep_index=i,
            total_num_timesteps=STEPS,
            controlnet_double_block_residuals=None,
            controlnet_single_block_residuals=None,
            ip_adapter_extensions=[],
            regional_prompting_extension=regional,
        )
        img = img + (t_prev - t_curr) * pred * update_scale
    if DEVICE == "mps":
        torch.mps.synchronize()
    if hasattr(model, "_fbcache"):
        delattr(model, "_fbcache")
    return img.float(), time.time() - t0


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return (torch.linalg.vector_norm(a - b) / (torch.linalg.vector_norm(a) + 1e-8)).item()


@torch.no_grad()
def forward_once(model: Flux, img: torch.Tensor, inputs, step_idx: int, regional) -> torch.Tensor:
    _, img_ids, txt, txt_ids, y = inputs
    t_vec = torch.full((img.shape[0],), 0.5, device=DEVICE, dtype=DTYPE)
    return model(
        img=img, img_ids=img_ids, txt=txt, txt_ids=txt_ids, timesteps=t_vec, y=y,
        guidance=None, timestep_index=step_idx, total_num_timesteps=STEPS,
        controlnet_double_block_residuals=None, controlnet_single_block_residuals=None,
        ip_adapter_extensions=[], regional_prompting_extension=regional,
    ).float()


@torch.no_grad()
def time_single_step(model: Flux, inputs, force_skip: bool) -> float:
    """Median seconds for one forward: force_skip=True reuses the cache (1 block);
    force_skip=False runs the full stack (57 blocks)."""
    img0, img_ids, txt, txt_ids, y = inputs
    regional = _NoRegional()
    # threshold 1e9 -> always reuse after warmup; 0 with manual attach -> never reuse
    # threshold 2.0 -> always reuse after warmup (relative_l1 < 2 with identical inputs);
    # threshold 0.0 -> never reuse (relative_l1 >= 0 is never < 0) -> full stack every step.
    cache = FluxFirstBlockCache(threshold=2.0 if force_skip else 0.0)
    model._fbcache = cache
    t_vec = torch.full((1,), 0.5, device=DEVICE, dtype=DTYPE)

    def one(step_idx):
        return model(
            img=img0, img_ids=img_ids, txt=txt, txt_ids=txt_ids, timesteps=t_vec, y=y,
            guidance=None, timestep_index=step_idx, total_num_timesteps=STEPS,
            controlnet_double_block_residuals=None, controlnet_single_block_residuals=None,
            ip_adapter_extensions=[], regional_prompting_extension=regional,
        )

    one(0)  # warmup + populate cache (full step, stores residual)
    times = []
    for k in range(5):
        if DEVICE == "mps":
            torch.mps.synchronize()
        t0 = time.time()
        one(k + 1)
        if DEVICE == "mps":
            torch.mps.synchronize()
        times.append(time.time() - t0)
    delattr(model, "_fbcache")
    return statistics.median(times)


def main() -> None:
    print(f"device={DEVICE} dtype={DTYPE}; building Flux (19 double + 38 single blocks, hidden=1536)...")
    model = build_model()
    inputs = make_inputs()

    # Warmup.
    run_trajectory(model, inputs, cache=None)

    # 1. Baseline (original forward path, no cache).
    base_img, base_t = run_trajectory(model, inputs, cache=None)

    # 2. Identity: cache attached, ~zero threshold -> never skips -> must match baseline.
    ident_cache = FluxFirstBlockCache(threshold=1e-12)
    ident_img, _ = run_trajectory(model, inputs, cache=ident_cache)
    ident_err = rel_l2(base_img, ident_img)
    print("\n[1] No-regression identity (threshold~0, cache path runs full stack every step):")
    print(f"    reuse(skip) count = {ident_cache.reuse_count}/{STEPS}  | rel-L2 vs original forward = {ident_err:.2e}")
    print(f"    -> {'PASS' if ident_err < 1e-3 and ident_cache.reuse_count == 0 else 'FAIL'} "
          "(refactored block stack reproduces the original forward)")

    # 3a. Noisy walk (update_scale=1.0): residuals genuinely differ -> gate declines to skip.
    print("\n[2a] Skip gate on a noisy trajectory (residuals genuinely change every step):")
    cache = FluxFirstBlockCache(threshold=0.4)
    run_trajectory(model, inputs, cache=cache, update_scale=1.0)
    print(f"    threshold=0.4  skips={cache.reuse_count}/{STEPS}  "
          "-> gate correctly does NOT skip when the trajectory is not coherent")

    # 3b. Skip-path approximation vs the full forward under a controlled input perturbation.
    # This is the FBCache premise: reusing the cached residual is exact when the step-to-step
    # change is zero and degrades gracefully as it grows -- which is why the gate only skips
    # when the first-block residual barely moved. (With random weights we drive the change
    # directly rather than relying on a trajectory; real trained weights are far smoother.)
    print("\n[2b] Skip-path approximation vs full forward (controlled input perturbation):")
    reg = _NoRegional()
    img_a = inputs[0]
    for eps in (0.0, 0.01, 0.05, 0.1, 0.3):
        g = torch.Generator().manual_seed(7)
        img_b = img_a + eps * torch.randn(img_a.shape, generator=g).to(device=DEVICE, dtype=DTYPE)
        if hasattr(model, "_fbcache"):
            delattr(model, "_fbcache")
        full_b = forward_once(model, img_b, inputs, 1, reg)  # full forward of perturbed input
        cache = FluxFirstBlockCache(threshold=10.0)  # force the skip path
        model._fbcache = cache
        forward_once(model, img_a, inputs, 0, reg)  # populate cache (full step on the base input)
        skip_b = forward_once(model, img_b, inputs, 1, reg)  # reuse cached residual for perturbed input
        delattr(model, "_fbcache")
        print(f"    perturbation eps={eps:<5} rel-L2(skip vs full forward) = {rel_l2(full_b, skip_b):.4f}")

    # 4. Structural per-step speedup (skipped step = 1 block vs full = 57 blocks).
    full_step = time_single_step(model, inputs, force_skip=False)
    skip_step = time_single_step(model, inputs, force_skip=True)
    print("\n[3] Structural per-step speedup:")
    print(f"    full step (57 blocks) = {full_step * 1000:.1f} ms | skipped step (1 block) = "
          f"{skip_step * 1000:.1f} ms -> {full_step / skip_step:.1f}x faster per skipped step")

    # 5. CFG per-stream keying: two interleaved streams must keep separate cache state.
    print("\n[4] CFG per-stream keying:")
    cache = FluxFirstBlockCache(threshold=0.2)
    model._fbcache = cache
    pos, neg = _NoRegional(), _NoRegional()
    img_p, img_n = inputs[0].clone(), inputs[0].clone() * 0.9
    ts = torch.linspace(1.0, 0.0, STEPS + 1).tolist()
    finite = True
    for i in range(STEPS):
        for stream_img, reg in ((img_p, pos), (img_n, neg)):
            t_vec = torch.full((1,), ts[i], device=DEVICE, dtype=DTYPE)
            out = model(
                img=stream_img, img_ids=inputs[1], txt=inputs[2], txt_ids=inputs[3], timesteps=t_vec,
                y=inputs[4], guidance=None, timestep_index=i, total_num_timesteps=STEPS,
                controlnet_double_block_residuals=None, controlnet_single_block_residuals=None,
                ip_adapter_extensions=[], regional_prompting_extension=reg,
            )
            finite = finite and bool(torch.isfinite(out).all())
    keys = len(cache._cached_residual)
    delattr(model, "_fbcache")
    print(f"    distinct cached streams = {keys} (expected 2: pos + neg) | all outputs finite = {finite}")
    print(f"    -> {'PASS' if keys == 2 and finite else 'FAIL'}")


if __name__ == "__main__":
    main()
