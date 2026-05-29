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

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        # Per-stream previous first-block residual (drives the "did it change?" test).
        self._prev_first_residual: dict[int, torch.Tensor] = {}
        # Per-stream cached aggregate residual (everything after the first block).
        self._cached_residual: dict[int, torch.Tensor] = {}

    def get_cached_residual(self, stream_key: int) -> torch.Tensor | None:
        return self._cached_residual.get(stream_key)

    def should_reuse(self, stream_key: int, first_residual: torch.Tensor) -> bool:
        """True if `first_residual` is close enough to the previous step's that the cached
        aggregate residual may be reused (relative-L1 distance below the threshold)."""
        prev = self._prev_first_residual.get(stream_key)
        if prev is None or prev.shape != first_residual.shape:
            return False
        denom = prev.abs().mean()
        if not (denom > 0):
            return False
        relative_l1 = ((first_residual - prev).abs().mean() / denom).item()
        return relative_l1 < self.threshold

    def update_first_residual(self, stream_key: int, first_residual: torch.Tensor) -> None:
        """Record this step's first-block residual for next step's change test (every step)."""
        self._prev_first_residual[stream_key] = first_residual

    def store_residual(self, stream_key: int, residual: torch.Tensor) -> None:
        """Cache the aggregate residual computed on a full (non-skipped) step."""
        self._cached_residual[stream_key] = residual


@contextlib.contextmanager
def apply_first_block_cache(model: torch.nn.Module, threshold: float | None) -> Iterator[None]:
    """Attach a FluxFirstBlockCache to `model` for the duration of a single denoise run.

    No-op (and zero overhead in ``Flux.forward``) when ``threshold`` is falsy or <= 0. The
    cache lives on the private ``_fbcache`` attribute that ``Flux.forward`` checks, and is
    always detached on exit so it never leaks across runs -- every run must start clean.
    """
    if not threshold or threshold <= 0.0:
        yield
        return

    previous = getattr(model, "_fbcache", None)
    model._fbcache = FluxFirstBlockCache(threshold)
    try:
        yield
    finally:
        if previous is None:
            if hasattr(model, "_fbcache"):
                delattr(model, "_fbcache")
        else:
            model._fbcache = previous
