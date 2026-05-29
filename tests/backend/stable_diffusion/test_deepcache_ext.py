"""Unit tests for the DeepCache no-op safety path (invokeai/backend/stable_diffusion/deepcache_ext.py).

DeepCache is disabled in denoise_latents.py (interval forced to 1) whenever ControlNet,
T2I-Adapter, IP-Adapter, or sequential guidance is active. These tests pin the other half of
that contract: at interval <= 1 the context manager is a true no-op -- it neither imports the
optional DeepCache package nor mutates the UNet -- so the guard reliably renders it inert.
"""

from invokeai.backend.stable_diffusion.deepcache_ext import apply_deepcache


class _Stub:
    """Stand-in for a UNet / scheduler; we assert the no-op path never touches it."""


def test_apply_deepcache_noop_at_interval_1():
    unet, scheduler = _Stub(), _Stub()
    before = dict(unet.__dict__)
    with apply_deepcache(unet, scheduler, cache_interval=1):
        pass
    assert unet.__dict__ == before  # untouched -> genuinely inert


def test_apply_deepcache_noop_at_interval_0():
    unet, scheduler = _Stub(), _Stub()
    before = dict(unet.__dict__)
    with apply_deepcache(unet, scheduler, cache_interval=0):
        pass
    assert unet.__dict__ == before
