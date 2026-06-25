# Initially pulled from https://github.com/black-forest-labs/flux

from typing import Callable, Optional

import torch
from einops import rearrange
from torch import Tensor

# Optional fused Metal flash-attention kernel for the UNMASKED attention sites, installed by
# configure_flux_flash_attention() at denoise start. None => stock SDPA (the default). This is the
# per-call-site dispatcher hook: masked sites (regional prompting / IP-Adapter) always use stock SDPA
# because the available MPS flash kernels do not accept arbitrary attention masks.
_FUSED_ATTN: Optional[Callable[[Tensor, Tensor, Tensor], Tensor]] = None


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, attn_mask: Tensor | None = None) -> Tensor:
    q, k = apply_rope(q, k, pe)

    # apply_rope's view()-based reconstruction can leave q/k non-contiguous, which pushes the MPS
    # SDPA kernel onto a slower path; a contiguous Q/K keeps it on the fast path. Behavior-preserving.
    q, k = q.contiguous(), k.contiguous()

    if _FUSED_ATTN is not None and attn_mask is None:
        x = _FUSED_ATTN(q, k, v)  # fused kernel, unmasked sites only
    else:
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    # Native transpose+reshape (equivalent to einops "B H L D -> B L (H D)"), avoids the einops call
    # in the per-block hot path.
    x = x.transpose(1, 2).reshape(x.shape[0], x.shape[2], -1)

    return x


def _flash_parity_ok(fn: Callable[[Tensor, Tensor, Tensor], Tensor], tol: float = 2e-2) -> bool:
    """A fused kernel is only accepted if it numerically matches stock SDPA on a fixed probe input."""
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    dt = torch.bfloat16 if dev == "mps" else torch.float32
    g = torch.Generator().manual_seed(0)
    q = torch.randn(1, 4, 64, 32, generator=g).to(dev, dt)
    k = torch.randn(1, 4, 64, 32, generator=g).to(dev, dt)
    v = torch.randn(1, 4, 64, 32, generator=g).to(dev, dt)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    try:
        got = fn(q, k, v)
    except Exception:
        return False
    return got.shape == ref.shape and torch.allclose(got.float(), ref.float(), atol=tol, rtol=tol)


def _load_fused_attn_kernel() -> Optional[Callable[[Tensor, Tensor, Tensor], Tensor]]:
    """Best-effort import of an optional MPS flash-attention kernel WITHOUT hard-depending on it.
    Returns a callable (q,k,v)->out in [B,H,L,D] layout, or None if no compatible kernel is present.
    Add adapters here as kernels are vetted with scripts/flash_attn_probe.py."""
    # Intentionally no third-party kernel is bundled; this stays a no-op until one is vetted and
    # installed in the environment. Keeping the loader here makes activation a one-line change.
    return None


def configure_flux_flash_attention(enabled: bool, logger=None) -> bool:
    """Enable the fused flash-attention fast path for unmasked FLUX attention, if available & correct.

    Safe by construction: with `enabled=False` (the default) or no installed kernel, this leaves stock
    SDPA in place. A kernel is only activated after passing a numerical parity check vs SDPA.
    """
    global _FUSED_ATTN
    _FUSED_ATTN = None
    if not enabled:
        return False
    fn = _load_fused_attn_kernel()
    if fn is None:
        if logger is not None:
            logger.info("FLUX flash-attention requested, but no vetted MPS kernel is installed; using stock SDPA.")
        return False
    if not _flash_parity_ok(fn):
        if logger is not None:
            logger.warning("FLUX flash-attention kernel failed the parity check; using stock SDPA.")
        return False
    _FUSED_ATTN = fn
    if logger is not None:
        logger.info("FLUX flash-attention enabled for unmasked attention sites.")
    return True


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
