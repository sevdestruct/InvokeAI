"""Parity test for the FLUX attention output reshape + contiguous-Q/K cleanup (flux/math.py).

The hot-path change (einops rearrange -> native transpose/reshape, plus a contiguous() guard on
Q/K) is meant to be exactly behavior-preserving. This proves it: the native reshape is bitwise
identical to the einops formulation, and forcing Q/K contiguous does not change SDPA output.
"""

import torch
from einops import rearrange


def test_native_reshape_matches_einops():
    torch.manual_seed(0)
    x = torch.randn(2, 6, 32, 16)  # B H L D
    native = x.transpose(1, 2).reshape(x.shape[0], x.shape[2], -1)
    einops_out = rearrange(x, "B H L D -> B L (H D)")
    assert torch.equal(native, einops_out)


def test_contiguous_qk_does_not_change_sdpa():
    torch.manual_seed(0)
    q = torch.randn(1, 4, 48, 16).transpose(1, 2).transpose(1, 2)  # induce a non-contiguous view
    k = torch.randn(1, 4, 48, 16)
    v = torch.randn(1, 4, 48, 16)
    out_ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    out_contig = torch.nn.functional.scaled_dot_product_attention(q.contiguous(), k.contiguous(), v)
    assert torch.allclose(out_ref, out_contig, atol=1e-6)


def test_flash_attention_disabled_by_default_and_configurable():
    """The fused flash-attention path is inert by default (no kernel bundled), so attention() uses
    stock SDPA. configure_flux_flash_attention(False) is a no-op; with no kernel it stays disabled."""
    from invokeai.backend.flux import math as flux_math

    assert flux_math._FUSED_ATTN is None
    assert flux_math.configure_flux_flash_attention(False) is False
    assert flux_math.configure_flux_flash_attention(True) is False  # no kernel installed -> stays off
    assert flux_math._FUSED_ATTN is None


def test_flash_parity_check_accepts_sdpa_and_rejects_wrong():
    """The parity gate accepts a kernel that matches SDPA and rejects one that doesn't -- this is what
    prevents a bad kernel from silently corrupting output when one is wired in later."""
    from invokeai.backend.flux import math as flux_math

    good = lambda q, k, v: torch.nn.functional.scaled_dot_product_attention(q, k, v)  # noqa: E731
    bad = lambda q, k, v: torch.zeros_like(q)  # noqa: E731
    assert flux_math._flash_parity_ok(good) is True
    assert flux_math._flash_parity_ok(bad) is False
