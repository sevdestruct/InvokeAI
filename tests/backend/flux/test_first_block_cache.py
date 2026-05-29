"""Unit tests for the FLUX FirstBlockCache integration (invokeai/backend/flux/first_block_cache.py
and the cache branch in Flux.forward).

These verify the two properties that matter for correctness:
  - the cache actually reuses the aggregate residual when the first-block residual is unchanged, and
  - the cache is bypassed (no reuse) whenever ControlNet or IP-Adapter residuals are active, since
    those inject into individual blocks and are incompatible with skipping them.

A tiny real Flux model is used (CPU, random weights). Feeding the same input on two consecutive
steps makes the first-block residual identical, so a reuse is deterministic when not guarded.
"""

import torch

from invokeai.backend.flux.first_block_cache import FluxFirstBlockCache, apply_first_block_cache
from invokeai.backend.flux.model import Flux, FluxParams

B, IMG_SEQ, TXT_SEQ = 1, 4, 2
IN_CH, HIDDEN, CTX_DIM, VEC_DIM = 8, 64, 16, 8


class _NoRegional:
    def get_double_stream_attn_mask(self, block_index: int):
        return None

    def get_single_stream_attn_mask(self, block_index: int):
        return None


class _StubIPAdapter:
    """Minimal IP-Adapter extension: the block processor calls run_ip_adapter(); no-op passthrough."""

    def __init__(self) -> None:
        self.called = False

    def run_ip_adapter(self, *, timestep_index, total_num_timesteps, block_index, block, img_q, img):
        self.called = True
        return img


def _build_tiny_flux() -> Flux:
    params = FluxParams(
        in_channels=IN_CH,
        vec_in_dim=VEC_DIM,
        context_in_dim=CTX_DIM,
        hidden_size=HIDDEN,
        mlp_ratio=2.0,
        num_heads=4,
        depth=2,
        depth_single_blocks=2,
        axes_dim=[4, 6, 6],  # sums to hidden_size // num_heads = 16
        theta=10_000,
        qkv_bias=True,
        guidance_embed=False,
    )
    torch.manual_seed(0)
    return Flux(params).eval()


def _base_inputs():
    torch.manual_seed(1)
    return {
        "img": torch.randn(B, IMG_SEQ, IN_CH),
        "img_ids": torch.zeros(B, IMG_SEQ, 3),
        "txt": torch.randn(B, TXT_SEQ, CTX_DIM),
        "txt_ids": torch.zeros(B, TXT_SEQ, 3),
        "timesteps": torch.full((B,), 0.5),
        "y": torch.randn(B, VEC_DIM),
        "guidance": None,
    }


@torch.no_grad()
def _forward(model, inputs, *, step, cn_double=None, cn_single=None, ip=None, regional=None):
    # The cache keys on the regional-extension object identity, so a run must reuse the SAME
    # object across steps (as the real denoise loop does for its pos/neg streams).
    return model(
        img=inputs["img"],
        img_ids=inputs["img_ids"],
        txt=inputs["txt"],
        txt_ids=inputs["txt_ids"],
        timesteps=inputs["timesteps"],
        y=inputs["y"],
        guidance=inputs["guidance"],
        timestep_index=step,
        total_num_timesteps=4,
        controlnet_double_block_residuals=cn_double,
        controlnet_single_block_residuals=cn_single,
        ip_adapter_extensions=ip or [],
        regional_prompting_extension=regional if regional is not None else _NoRegional(),
    )


def test_apply_first_block_cache_noop_when_disabled():
    model = _build_tiny_flux()
    with apply_first_block_cache(model, 0.0):
        assert getattr(model, "_fbcache", None) is None
    assert getattr(model, "_fbcache", None) is None


def test_apply_first_block_cache_attaches_and_detaches():
    model = _build_tiny_flux()
    with apply_first_block_cache(model, 0.1):
        assert isinstance(model._fbcache, FluxFirstBlockCache)
    # Always detached on exit so state never leaks into a later run.
    assert getattr(model, "_fbcache", None) is None


def test_fbcache_reuses_residual_when_first_block_unchanged():
    model = _build_tiny_flux()
    inputs = _base_inputs()
    cache = FluxFirstBlockCache(threshold=0.1)
    model._fbcache = cache
    regional = _NoRegional()

    _forward(model, inputs, step=0, regional=regional)  # populates the cache (full pass)
    out = _forward(model, inputs, step=1, regional=regional)  # identical input -> residual unchanged -> reuse

    assert cache.reuse_count == 1
    assert torch.isfinite(out).all()


def test_fbcache_bypassed_with_controlnet_residuals():
    model = _build_tiny_flux()
    inputs = _base_inputs()
    cache = FluxFirstBlockCache(threshold=0.1)
    model._fbcache = cache

    cn_double = [torch.zeros(B, IMG_SEQ, HIDDEN) for _ in range(len(model.double_blocks))]
    cn_single = [torch.zeros(B, IMG_SEQ, HIDDEN) for _ in range(len(model.single_blocks))]
    regional = _NoRegional()

    _forward(model, inputs, step=0, cn_double=cn_double, cn_single=cn_single, regional=regional)
    _forward(model, inputs, step=1, cn_double=cn_double, cn_single=cn_single, regional=regional)

    # ControlNet injects per-block residuals -> the cache branch must never be taken.
    assert cache.reuse_count == 0


def test_fbcache_bypassed_with_ip_adapter():
    model = _build_tiny_flux()
    inputs = _base_inputs()
    cache = FluxFirstBlockCache(threshold=0.1)
    model._fbcache = cache

    ip = [_StubIPAdapter()]
    regional = _NoRegional()
    _forward(model, inputs, step=0, ip=ip, regional=regional)
    _forward(model, inputs, step=1, ip=ip, regional=regional)

    # IP-Adapter injects per-block conditioning -> the cache branch must never be taken.
    assert cache.reuse_count == 0
    assert ip[0].called  # confirms the full (non-cached) block path actually ran
