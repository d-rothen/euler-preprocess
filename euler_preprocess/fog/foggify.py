"""Backward-compatibility shim.

The canonical locations are now:
- ``euler_preprocess.fog.transform.FogTransform`` (class)
- ``euler_preprocess.fog.models`` (standalone fog functions)
"""
from __future__ import annotations

# Re-export everything that existing code may import from this module.
from euler_preprocess.common.sampling import format_value  # noqa: F401
from euler_preprocess.fog.models import (  # noqa: F401
    AIRLIGHT_METHODS,
    DEFAULT_CONTRAST_THRESHOLD,
    DEFAULT_MODEL_CONFIGS,
    apply_fog,
    apply_fog_torch,
    apply_model,
    estimate_airlight_torch,
    modulate_with_noise,
    modulate_with_noise_torch,
    normalize_atmospheric_light,
    normalize_atmospheric_light_torch,
    resolve_model_config,
    resolve_scales,
    select_model,
    visibility_to_k,
)
from euler_preprocess.fog.transform import FogTransform, Foggify  # noqa: F401
