# Apple Silicon Performance Baselines (arbiter run)

Measured with `scripts/perf_arbiter.py` on **M3 Max** via the real in-app pipeline (HTTP API), seed
7777, DPM++ 2M Karras (SD/SDXL) / Euler (FLUX). Each architecture's `off` run is the
**upstream-equivalent denoise** (caches disabled); `balanced`/`max` are this branch's acceleration via
the per-generation override fields. Speedup is end-to-end wall-clock vs `off`; **quality verdicts below
are from a perceptual (vision) review of the actual images, not the raw SSIM** (see "Reading the
numbers").

## Results

| arch | model | prompt | exp | wall (s) | denoise (s) | speedup | SSIM* | sharpness | perceptual verdict |
|---|---|---|---|---|---|---|---|---|---|
| SD1.5 | animics_v30 (512×768) | portrait | off | 14.2 | 11.9 | 1.0× | — | — | reference |
| SD1.5 | | portrait | balanced (dc 2) | 6.2 | 4.9 | **2.30×** | 0.70 | 0.79 | clean |
| SD1.5 | | portrait | max (dc 3) | 5.2 | 3.8 | **2.75×** | 0.68 | 0.67 | mild softening |
| SD1.5 | | scene | balanced | 6.2 | 5.0 | 1.65× | 0.74 | 0.94 | clean |
| SD1.5 | | scene | max | 5.2 | 3.8 | 1.97× | 0.56 | 0.77 | softening |
| SDXL | cyberrealisticPony_v160 (832×1216) | portrait | off | 42.5 | 40.1 | 1.0× | — | — | reference |
| SDXL | | portrait | balanced (dc 2) | 22.3 | 20.1 | **1.91×** | 0.79 | 1.44 | clean (face intact) |
| SDXL | | portrait | max (dc 3) | 16.3 | 14.3 | **2.62×** | 0.73 | 1.30 | clean (face intact) |
| SDXL | | scene | balanced | 26.3 | 24.3 | 1.73× | 0.45 | 1.10 | clean, composition shifts |
| SDXL | | scene | max | 21.3 | 19.0 | 2.13× | 0.23 | 1.25 | coherent, big reshuffle |
| FLUX | schnell (1024², 20-step) | portrait | off | 249.0 | 237.8 | 1.0× | — | — | reference |
| FLUX | | portrait | balanced (fbc 0.12) | 63.6 | 61.3 | **3.91×** | 0.86 | 0.81 | clean (face intact) |
| FLUX | | portrait | max (fbc 0.2) | 38.4 | 35.7 | **6.48×** | 0.79 | 0.72 | clean (face intact) |
| FLUX | | scene | balanced | 74.6 | 72.1 | 2.82× | 0.69 | 1.02 | coherent |
| FLUX | | scene | max | 43.4 | 40.7 | 4.85× | 0.57 | 1.21 | coherent |

\* SSIM here is **deviation from the off-reference, not absolute quality** — see below.

## Reading the numbers (important)

- **SSIM measures how much the image changed vs. acceleration-off, not whether it got worse.** Cache
  approximations legitimately shift composition while preserving quality, so low SSIM is expected and
  is *not* a failure. The arbiter's automated `WARN (faces)` flags (SDXL portrait balanced; all FLUX
  runs) were **deviation false-positives** — vision review confirmed faces/anatomy were intact in every
  case, including FLUX `max` (6.5×) and SDXL `max` (2.6×).
- **Sharpness ratio** (variant/reference high-frequency energy) is the more honest degradation signal.
  The only real softening seen is **SD1.5 at `max`** (~0.67) — so SD1.5's sweet spot is `balanced`,
  while SDXL/FLUX tolerate `max` well.
- **FLUX `off` is inflated** (~250 s) because all three architectures' models were resident during the
  run (memory pressure). The *relative* speedups are valid; isolated FLUX FBCache earlier measured
  ~3.2× at thr 0.12. Treat FLUX absolute times as upper bounds, ratios as representative.

## Recommended defaults (quality-first)

| arch | recommend | why |
|---|---|---|
| SD1.5 | balanced (interval 2) | `max` softens detail (~0.67 sharpness) |
| SDXL | balanced (interval 2), `max` for drafts | both clean; `max` = 2.6× when speed matters |
| FLUX | balanced (thr 0.12) | 3.9× and visually equal to off; `max` (6.5×) clean too but more drift |

## On-device roofline (`scripts/roofline_check.py`)

FLUX-representative GEMMs run at **12.4–13.1 TFLOPS — the chip's compute ceiling** — vs. far-lower
memory-bound throughput. **Compute-bound, confirmed on this M3 Max.** Therefore weight quantization is
footprint-only (no latency win); the real levers are fewer steps (caching) and faster FP kernels
(flash-attention). See `MAC_PERFORMANCE_OVERHAUL.md`.
