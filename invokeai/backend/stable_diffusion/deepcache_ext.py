"""DeepCache integration for InvokeAI's modular SD/SDXL denoise loop.

DeepCache (https://github.com/horseee/DeepCache) skips the deep UNet blocks on most
timesteps, reusing their cached output and only recomputing the shallow layers. Because
it removes real compute (unlike token-merging, which only adds overhead on MPS), it
delivers a measured ~1.8-2.9x speedup on Apple Silicon at modest quality cost.

The upstream DeepCacheSDHelper reads the timestep positionally (args[1]) from a diffusers
pipeline call. InvokeAI calls the UNet with keyword args and runs its own loop, so we
subclass the helper to track the step with a simple per-call counter (InvokeAI issues one
batched cond+uncond UNet call per step). Everything else — the block-forward caching — is
reused unchanged, and is fully removed again on exit so the shared cached UNet is restored.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import invokeai.backend.util.logging as logger


@contextmanager
def apply_deepcache(unet, scheduler, cache_interval: int, cache_branch_id: int = 0) -> Iterator[None]:
    """Apply DeepCache to `unet` for the duration of denoising, then fully restore it.

    No-op when cache_interval <= 1 or when the optional `DeepCache` package is unavailable.

    Args:
        unet: A diffusers UNet2DConditionModel (SD 1.x / SDXL).
        scheduler: The active scheduler (kept for API parity; step tracking uses a counter).
        cache_interval: Recompute the deep blocks every Nth step; reuse cached output otherwise.
            1 disables caching.
        cache_branch_id: Which UNet branch/depth to cache from (0 = deepest, most aggressive).
    """
    if not cache_interval or cache_interval <= 1:
        yield
        return

    try:
        from DeepCache import DeepCacheSDHelper
    except ImportError:
        logger.warning("DeepCache requested but the `DeepCache` package is not installed (`pip install DeepCache`).")
        yield
        return

    class _InvokeDeepCacheHelper(DeepCacheSDHelper):
        # InvokeAI calls the UNet with kwargs and one batched call per step, so derive the
        # step index from a call counter rather than from a positional timestep argument.
        def reset_states(self):  # noqa: D102
            super().reset_states()
            self._call_counter = 0

        def wrap_unet_forward(self):  # noqa: D102
            self.function_dict["unet_forward"] = self.pipe.unet.forward

            def wrapped_forward(*args, **kwargs):
                self.cur_timestep = self._call_counter
                self._call_counter += 1
                return self.function_dict["unet_forward"](*args, **kwargs)

            self.pipe.unet.forward = wrapped_forward

    # DeepCacheSDHelper only touches pipe.unet and pipe.scheduler; a lightweight shim suffices.
    class _Shim:
        pass

    shim = _Shim()
    shim.unet = unet
    shim.scheduler = scheduler

    helper = _InvokeDeepCacheHelper(pipe=shim)
    helper.set_params(cache_interval=cache_interval, cache_branch_id=cache_branch_id)
    helper.enable()
    logger.info(f"DeepCache enabled (cache_interval={cache_interval}, cache_branch_id={cache_branch_id}).")
    try:
        yield
    finally:
        helper.disable()
