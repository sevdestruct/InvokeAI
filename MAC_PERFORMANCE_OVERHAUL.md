# Apple Silicon Performance Overhaul — Plan & Findings

Target hardware: **Apple M3 Max** (40-core GPU, 128 GB unified memory, ~400 GB/s), macOS 26,
PyTorch 2.7 MPS backend, InvokeAI (diffusers) for SD 1.5 / SDXL and FLUX.1.

This is the living plan for the senior graphics-optimization overhaul. It records what is already
shipped, the **verified** direction (separated from speculative), and the phased work ahead. Every
performance claim is gated by the arbiter harness (`scripts/perf_arbiter.py`) + perceptual review —
no number ships unmeasured.

---

## 0. The decisive finding (resolved by adversarial research)

**FLUX DiT denoise on M3 Max is compute-bound, not bandwidth-bound.** A first-principles roofline
(~4600 FLOP/byte for FLUX FFN GEMMs vs the M3 Max ridge of ~0.035 FLOP/byte), corroborated by the
Ideogram INT8 paper, SVDQuant, MLX, and mflux benchmark provenance, places every FLUX GEMM ~5 orders
of magnitude above the roofline ridge.

Consequences (these reorder the whole strategy):
- **Weight-only quantization (4-bit/fp8) is a footprint tool, NOT a speed lever** on M1–M4 (no integer
  tensor path; the only "4-bit faster" numbers are a 16 GB swap artifact that vanishes at 128 GB).
  Keep FLUX in bf16. *(Corrects an earlier-session assumption that FLUX was bandwidth-bound.)*
- **The only real speed levers are: (a) do less compute (step-skipping caches), and (b) faster FP
  attention/GEMM kernels.**
- **Metal 4 tensor ops give ~0% on M3 Max** (M5-only payoff; llama.cpp disabled them pre-M5).

→ Action item P1-D: confirm on-device with the MPS profiler / a compute-utilization microbench
(`scripts/roofline_check.py`, TODO) — expect high ALU occupancy, low memory stall.

---

## 1. Already shipped (on `mps-perf`)

| Area | What | Status |
|---|---|---|
| MPS startup | `PYTORCH_ENABLE_MPS_FALLBACK`, high-watermark ratio, friendly device name ("Apple M3 Max GPU (Metal/MPS)") | shipped |
| Model cache | Lifted the 32 GB cap on unified memory (≈62 GB) | shipped |
| DeepCache (SD/SDXL) | Opt-in deep-block reuse; `deepcache_interval`; guarded off under ControlNet/T2I/IP-Adapter/seq-guidance | shipped + tested |
| FLUX FirstBlockCache | Residual reuse/skip; `flux_first_block_cache_threshold`; CFG per-stream keyed; guarded off under ControlNet/IP-Adapter | shipped + tested |
| Per-generation overrides | `deepcache_interval` + `first_block_cache_threshold` node fields (live, no restart) | shipped + tested |
| Simple-UI control | "Acceleration: Off / Balanced / Max" on Canvas **and** Generate tabs | shipped |
| Arbiter harness | API-driven {arch × prompt × experiment} runner; SSIM/PSNR/MAD + face-region SSIM + sharpness + diff heatmaps; PASS/WARN/FAIL ledger | shipped |
| Bonus | Trackpad pinch/wheel zoom + pan + arrow-nav in the image viewer | shipped |

Measured (in-app, this branch): FLUX FBCache ≈ **3.2×** (thr 0.12, 20-step schnell) at ~2.4% MAD;
DeepCache SDXL ≈ **1.9×** (interval 2) / **2.6×** (interval 3), perceptually clean faces through
interval 3 (verified by vision review — see §4).

---

## 2. The validated compounding stack (build order)

These three occupy **different axes** and compound: faster attention kernel × fewer/cheaper steps
× better solver. Each is gated by the arbiter + an in-house perceptual check.

### Phase 1 — Free prerequisites (own code, low risk)
- **P1-A einops→native reshape + `.contiguous()` guard on Q** in the FLUX/SD attention path
  (`invokeai/backend/flux/math.py`, `custom_block_processor.py`, `custom_atttention.py`). Few %–10%
  on the attention path **and** fixes a torch 2.8 SDPA stride correctness bug. Verify bitwise-close
  output parity before/after.
- **P1-D roofline confirmation microbench** — bank the compute-bound finding on this exact chip.

### Phase 2 — Per-call-site Metal flash-attention (the #1 ship-grade win, ~15–25% end-to-end)
PyTorch-MPS routes SDPA through the MPSGraph "math" path and never calls a fused flash kernel
(pytorch #179294). A real fused Metal flash-attention (mps-flash-attn / kernels-community
metal-flash-sdpa) beats it. **Draw Things shows up to ~20% on FLUX (M3/M4).**

Hard constraints (verified — ignoring them silently corrupts output):
- **Per-call-site dispatcher, NEVER a global `replace_sdpa()` monkeypatch** (that hijacks VAE +
  text-encoder, and the kernels silently drop masks).
- **Gate on `attn_mask is None`.** Masked sites stay on stock SDPA: FLUX boolean regional mask
  (`regional_prompting_extension.py:206`), SD additive-float regional (`custom_atttention.py:96`).
- Eligible unmasked sites: FLUX `math.py:11`; IP-Adapter cross-attn (`xlabs_ip_adapter_extension.py:83`,
  `ip_double_stream_block_processor.py:86`, `custom_atttention.py:183`); `custom_atttention.py:129`
  when `attention_mask is None`.
- Add a **startup numerical-parity self-check** (fixed Q/K/V, assert max-abs-diff < tol) + a config
  toggle + auto-fallback.
- **DECISION POINT:** this introduces a third-party Metal kernel dependency. Needs explicit OK before
  adding (consistent with this project's "isolate experimental deps" rule). Watch-item: when
  pytorch #179294 lands `sdpa_full_attention_mps`, drop the third-party kernel.

### Phase 3 — Error-bounded caching (MagCache) validated in-house
MagCache (magnitude error-modeled cache) is the error-bounded successor to FBCache/TaylorSeer on the
**same temporal axis** (it replaces, not stacks). It needs **one calibration table per checkpoint**
(ships like diffusers `FLUX_MAG_RATIOS`), transfers across step counts/schedulers.
- **Quality is the gating deliverable, not speed.** Every published cache number (MagCache "near
  lossless", SpeCa "5.5%") is author-reported CUDA; MagCache is really LPIPS 0.20–0.26 vs baseline.
  **No independent perceptual benchmark exists at 1024px** — our arbiter (+ an added ImageReward/LPIPS
  metric) is exactly the missing gate. Default to the conservative threshold; sweep guidance scale.

### Phase 4 — DPM-Solver-v3 (orthogonal, composes with caching)
Register as a scheduler option; A/B at matched step budgets (28→20) on the perceptual harness.
1.3–2× at matched quality in the non-distilled regime.

---

## 3. High-ceiling bets (research-grade) & dead-ends

**Bets (spike only after Phase 2/3 measured):** SpeCa as a "Max" preset (calibration-free, but hides
a ~12% GenEval drop, needs MPS port); ClusCa (spatial token clustering — the one method on a *different*
axis that could multiply with a temporal cache); VAE-decode/text-encode overlap on unified memory.

**Verified dead-ends — do NOT pursue on M3 Max:**
- Weight-only quantization *for speed* (compute-bound; footprint only; irrelevant at 128 GB).
- Metal 4 tensor ops / ML encoder (M5-only; needs frozen CoreML graph; no PyTorch MTLTensor bridge).
- Global `replace_sdpa()` monkeypatch (corrupts masked attention).
- bghira universal-metal-flash-attention PyTorch op (single-head + causal-only; ignores masks).
- CUDA-only stacks: SVDQuant/Nunchaku, Chipmunk, SpargeAttention, torchao-on-MPS, CUDA-graph replay,
  torch.compile/Inductor-on-MPS, stable-diffusion.cpp/ggml.
- TaylorSeer naive linear forecast on top of threshold-gated FBCache (tried this session → artifacts).

---

## 4. Validation methodology (the arbiter)

`scripts/perf_arbiter.py` runs the **real in-app pipeline** over the API across {SD1.5/SDXL/FLUX ×
prompts × off/balanced/max}, and for each variant vs the acceleration-off reference records: wall
clock, per-stage timing, cache skip-count, peak cache memory, and MAD/PSNR/SSIM + face-region SSIM +
sharpness ratio + a diff heatmap. Verdicts are **artifact-based** (degenerate/blur/face-degradation),
because SSIM measures *deviation* and an approximation cache legitimately shifts composition.

**Critical lesson (this session):** numeric deviation ≠ quality. SDXL portrait at interval 2 scored
low face-region SSIM (`WARN`), but vision review showed the face fully intact — a deviation false
positive. **The arbiter localizes where to look; a perceptual pass (vision model / human) is the final
judge.** Planned upgrade: add ImageReward + LPIPS, and an optional vision-model perceptual verdict per
flagged run.

---

## 5. The "no GPU" question — answered

There is **no "no GPU" in the backend**. InvokeAI uses the M3 Max GPU via MPS: it logs `Using torch
device: ...`, loads every model `onto mps device`, and matmul runs at **12.3 TFLOPS fp16** (near the
chip's peak). The confusion came from (a) the bare device string `MPS` (now relabeled to
`Apple M3 Max GPU (Metal/MPS)`), and (b) the `MPS: enabled CPU fallback` line, which is a *robustness*
setting (unsupported ops fall back to CPU), not "CPU-only". The GPU is fully engaged; the path to
"faster" is §2, not "turn on the GPU".
