import logging
import os
import sys


def configure_torch_mps(enable_fallback: bool, high_watermark_ratio: float | None, logger: logging.Logger) -> None:
    """Configure Apple Silicon / MPS-related environment variables.

    These variables are read by torch when it is first imported, so this MUST be called before torch is imported
    (mirroring configure_torch_cuda_allocator). It is a no-op on non-macOS platforms.

    Args:
        enable_fallback: If True, set PYTORCH_ENABLE_MPS_FALLBACK=1 so ops not implemented for MPS fall back to CPU
            instead of raising. This trades a slow CPU path for robustness (no crashes on unsupported ops).
        high_watermark_ratio: If set, exported as PYTORCH_MPS_HIGH_WATERMARK_RATIO. 0.0 disables the upper bound on
            MPS allocations (use with caution). Leave None to keep the PyTorch default.
        logger: Logger for status messages.
    """
    if sys.platform != "darwin":
        return

    if "torch" in sys.modules:
        raise RuntimeError("configure_torch_mps() must be called before importing torch.")

    if enable_fallback and "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ:
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        logger.info("MPS: enabled CPU fallback for unsupported ops (PYTORCH_ENABLE_MPS_FALLBACK=1).")

    if high_watermark_ratio is not None and "PYTORCH_MPS_HIGH_WATERMARK_RATIO" not in os.environ:
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = str(high_watermark_ratio)
        logger.info(f"MPS: set PYTORCH_MPS_HIGH_WATERMARK_RATIO={high_watermark_ratio}.")
