"""Contract tests for the per-generation cache-override fields on the denoise invocations.

DeepCache (SD/SDXL) and FLUX FirstBlockCache are configured globally via invokeai.yaml. These
optional node fields let a single generation override the app setting (e.g. to A/B quality vs.
speed) without a restart. The fields MUST default to None so that callers which don't set them
(the linear UI graph builders) fall back to the config value -- preserving existing behavior.
"""

import pydantic
import pytest

from invokeai.app.invocations.denoise_latents import DenoiseLatentsInvocation
from invokeai.app.invocations.flux_denoise import FluxDenoiseInvocation


def _constraints(field):
    return {type(m).__name__: getattr(m, type(m).__name__.lower(), None) for m in field.metadata}


def test_deepcache_interval_field_defaults_to_none():
    # None -> the invocation falls back to get_config().deepcache_interval (no behavior change).
    assert DenoiseLatentsInvocation.model_fields["deepcache_interval"].default is None


def test_first_block_cache_threshold_field_defaults_to_none():
    assert FluxDenoiseInvocation.model_fields["first_block_cache_threshold"].default is None


def test_deepcache_interval_must_be_at_least_1():
    assert "Ge" in _constraints(DenoiseLatentsInvocation.model_fields["deepcache_interval"])


def test_first_block_cache_threshold_is_bounded_0_to_1():
    c = _constraints(FluxDenoiseInvocation.model_fields["first_block_cache_threshold"])
    assert "Ge" in c and "Le" in c


@pytest.mark.parametrize("interval", [-1, 0])
def test_deepcache_interval_rejects_below_1(interval):
    with pytest.raises(pydantic.ValidationError):
        DenoiseLatentsInvocation(id="t", deepcache_interval=interval)


@pytest.mark.parametrize("threshold", [-0.1, 1.5])
def test_first_block_cache_threshold_rejects_out_of_range(threshold):
    with pytest.raises(pydantic.ValidationError):
        FluxDenoiseInvocation(id="t", first_block_cache_threshold=threshold)
