# Initially pulled from https://github.com/black-forest-labs/flux

import torch
from einops import rearrange
from torch import Tensor


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, attn_mask: Tensor | None = None) -> Tensor:
    q, k = apply_rope(q, k, pe)

    # apply_rope's view()-based reconstruction can leave q/k non-contiguous, which pushes the MPS
    # SDPA kernel onto a slower path; a contiguous Q/K keeps it on the fast path. Behavior-preserving.
    q, k = q.contiguous(), k.contiguous()

    x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    # Native transpose+reshape (equivalent to einops "B H L D -> B L (H D)"), avoids the einops call
    # in the per-block hot path.
    x = x.transpose(1, 2).reshape(x.shape[0], x.shape[2], -1)

    return x


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = (
        torch.arange(0, dim, 2, dtype=torch.float32 if pos.device.type == "mps" else torch.float64, device=pos.device)
        / dim
    )
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.to(dtype=pos.dtype, device=pos.device)


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.view(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.view(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.view(*xq.shape).type_as(xq), xk_out.view(*xk.shape).type_as(xk)
