# FLUX FirstBlockCache: residual-reuse acceleration for the FLUX transformer.
#
# Technique reference: "First Block Cache" (https://github.com/chengzeyi/ParaAttention),
# a training-free DiT inference accelerator closely related to TeaCache. It exploits the
# observation that the change in the *first* transformer block's output between consecutive
# denoising steps is a strong predictor of the change in the *whole* transformer's output.
# On steps where the first block's residual is close (in relative L1) to the previous step's,
# every remaining block is skipped and a cached aggregate residual is reused instead.

from __future__ import annotations

import contextlib
from typing import Iterator

import torch


class FluxFirstBlockCache:
    """Per-stream residual cache backing the FLUX FirstBlockCache acceleration.

    Classifier-free guidance runs the transformer twice per step (positive + negative) with
    different conditioning, so the cache is keyed per stream -- the caller passes a distinct
    key for each conditioning stream -- to avoid cross-contaminating the two trajectories.
    """

    def __init__(self, threshold: float, error_budget: float = 0.0) -> None:
        self.threshold = threshold
        # Optional error-bounded mode (TeaCache/MagCache family). When > 0, instead of gating on the
        # instantaneous first-block change, accumulate the relative-L1 change across steps and only
        # recompute when the accumulated change exceeds the budget -- this BOUNDS the total drift from
        # a run of consecutive skips, which a fixed instantaneous threshold does not.
        self.error_budget = error_budget
        # Number of block-stack evaluations skipped via cache reuse (telemetry).
        self.reuse_count = 0
        # Per-stream previous first-block residual (drives the "did it change?" test).
        self._prev_first_residual: dict[int, torch.Tensor] = {}
        # Per-stream cached aggregate residual (everything after the first block).
        self._cached_residual: dict[int, torch.Tensor] = {}
        # Per-stream accumulated relative-L1 change since the last full compute (error-bounded mode).
        self._accum_error: dict[int, float] = {}

    def get_cached_residual(self, stream_key: int) -> torch.Tensor | None:
        return self._cached_residual.get(stream_key)

    def should_reuse(self, stream_key: int, first_residual: torch.Tensor) -> bool:
        """Decide whether the cached aggregate residual may be reused this step.

        Default (error_budget == 0): reuse when the instantaneous relative-L1 change of the first-block
        residual is below `threshold`. Error-bounded mode (error_budget > 0): accumulate that relative-L1
        change and reuse while the accumulation stays under the budget, forcing a full recompute (and
        resetting the accumulator) once it is exceeded -- bounding total drift across consecutive skips.
        """
        prev = self._prev_first_residual.get(stream_key)
        if prev is None or prev.shape != first_residual.shape:
            return False
        denom = prev.abs().mean()
        if not (denom > 0):
            return False
        relative_l1 = ((first_residual - prev).abs().mean() / denom).item()
        if self.error_budget > 0:
            accum = self._accum_error.get(stream_key, 0.0) + relative_l1
            reuse = accum < self.error_budget
            self._accum_error[stream_key] = accum if reuse else 0.0  # reset on a forced recompute
        else:
            reuse = relative_l1 < self.threshold
        if reuse:
            self.reuse_count += 1
        return reuse

    def update_first_residual(self, stream_key: int, first_residual: torch.Tensor) -> None:
        """Record this step's first-block residual for next step's change test (every step)."""
        self._prev_first_residual[stream_key] = first_residual

    def store_residual(self, stream_key: int, residual: torch.Tensor) -> None:
        """Cache the aggregate residual computed on a full (non-skipped) step."""
        self._cached_residual[stream_key] = residual


@contextlib.contextmanager
def apply_first_block_cache(
    model: torch.nn.Module, threshold: float | None, error_budget: float = 0.0, logger=None
) -> Iterator[None]:
    """Attach a FluxFirstBlockCache to `model` for the duration of a single denoise run.

    No-op (and zero overhead in ``Flux.forward``) when both ``threshold`` and ``error_budget`` are
    falsy/<= 0. The cache lives on the private ``_fbcache`` attribute that ``Flux.forward`` checks, and
    is always detached on exit so it never leaks across runs -- every run must start clean. When
    `logger` is provided, the number of skipped block-stack evaluations is logged.
    """
    if (not threshold or threshold <= 0.0) and (not error_budget or error_budget <= 0.0):
        yield
        return

    previous = getattr(model, "_fbcache", None)
    cache = FluxFirstBlockCache(threshold or 0.0, error_budget=error_budget or 0.0)
    model._fbcache = cache
    try:
        yield
    finally:
        if previous is None:
            if hasattr(model, "_fbcache"):
                delattr(model, "_fbcache")
        else:
            model._fbcache = previous
        if logger is not None and cache.reuse_count:
            mode = f"error_budget={error_budget}" if error_budget and error_budget > 0 else f"threshold={threshold}"
            logger.info(
                f"FLUX FirstBlockCache ({mode}): skipped the block stack on "
                f"{cache.reuse_count} step evaluation(s) by reusing the cached residual."
            )
