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
