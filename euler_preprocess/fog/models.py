from __future__ import annotations

import logging
import math

import numpy as np

from euler_preprocess.common.noise import perlin_fbm
from euler_preprocess.common.sampling import deep_merge, sample_value

try:
    import torch
except ImportError:
    torch = None

_logger = logging.getLogger("foggify")

AIRLIGHT_METHODS = ("from_sky", "dcp", "dcp_heuristic")

DEFAULT_CONTRAST_THRESHOLD = 0.05

DEFAULT_MODEL_CONFIGS = {
    "uniform": {
        "visibility_m": {"dist": "constant", "value": 80.0},
        "atmospheric_light": "from_sky",
    },
    "heterogeneous_k": {
        "visibility_m": {"dist": "constant", "value": 80.0},
        "atmospheric_light": "from_sky",
        "k_hetero": {
            "scales": "auto",
            "min_scale": 2,
            "max_scale": None,
            "min_factor": 0.0,
            "max_factor": 1.0,
            "normalize_to_mean": True,
        },
    },
    "heterogeneous_ls": {
        "visibility_m": {"dist": "constant", "value": 80.0},
        "atmospheric_light": "from_sky",
        "ls_hetero": {
            "scales": "auto",
            "min_scale": 2,
            "max_scale": None,
            "min_factor": 0.0,
            "max_factor": 1.0,
            "normalize_to_mean": False,
        },
    },
    "heterogeneous_k_ls": {
        "visibility_m": {"dist": "constant", "value": 80.0},
        "atmospheric_light": "from_sky",
        "k_hetero": {
            "scales": "auto",
            "min_scale": 2,
            "max_scale": None,
            "min_factor": 0.0,
            "max_factor": 1.0,
            "normalize_to_mean": True,
        },
        "ls_hetero": {
            "scales": "auto",
            "min_scale": 2,
            "max_scale": None,
            "min_factor": 0.0,
            "max_factor": 1.0,
            "normalize_to_mean": False,
        },
    },
}


def visibility_to_k(visibility_m: float, contrast_threshold: float) -> float:
    if visibility_m <= 0:
        raise ValueError(f"Visibility must be > 0, got {visibility_m}")
    return -math.log(contrast_threshold) / visibility_m


def normalize_atmospheric_light(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.ndim == 0:
        value = np.array([value, value, value], dtype=np.float32)
    if value.ndim != 1 or value.shape[0] != 3:
        raise ValueError("atmospheric_light must be scalar or length-3 list")
    if value.max() > 1.0:
        value = value / 255.0
    return np.clip(value, 0.0, 1.0)


def normalize_atmospheric_light_torch(value: "torch.Tensor") -> "torch.Tensor":
    value_t = value.to(torch.float32)
    if value_t.ndim == 0:
        value_t = value_t.repeat(3)
    if value_t.ndim == 1:
        value_t = value_t.unsqueeze(0)
    if value_t.ndim != 2 or value_t.shape[-1] != 3:
        raise ValueError("atmospheric_light must be scalar or length-3 list")
    max_val = float(value_t.max().item()) if value_t.numel() else 0.0
    if max_val > 1.0:
        value_t = value_t / 255.0
    return torch.clamp(value_t, 0.0, 1.0)


def estimate_airlight_torch(
    image: "torch.Tensor",
    sky_mask: "torch.Tensor",
    sample_id: str | None = None,
) -> "torch.Tensor":
    if sky_mask.sum() == 0:
        id_str = f" (sample {sample_id})" if sample_id else ""
        _logger.warning(
            "No sky pixels in segmentation mask%s; "
            "using default airlight fallback [1.0, 1.0, 1.0]",
            id_str,
        )
        return torch.ones(3, device=image.device, dtype=image.dtype)
    airlight_pixels = image[sky_mask]
    airlight = airlight_pixels.mean(dim=0)
    if not torch.all(torch.isfinite(airlight)):
        id_str = f" (sample {sample_id})" if sample_id else ""
        _logger.warning(
            "Airlight estimated from sky pixels contains non-finite values "
            "(%s)%s; using default airlight fallback [1.0, 1.0, 1.0]",
            airlight.tolist(),
            id_str,
        )
        return torch.ones(3, device=image.device, dtype=image.dtype)
    return airlight


def resolve_scales(
    hetero_cfg: dict, height: int, width: int, rng: np.random.Generator
) -> list[int]:
    scales_spec = hetero_cfg.get("scales", "auto")
    scales_spec = sample_value(scales_spec, rng)
    if isinstance(scales_spec, str):
        if scales_spec != "auto":
            raise ValueError(f"Unsupported scales value: {scales_spec}")
        min_scale = int(sample_value(hetero_cfg.get("min_scale", 2), rng))
        max_scale = hetero_cfg.get("max_scale", None)
        if max_scale is None:
            max_scale = max(height, width)
        max_scale = int(sample_value(max_scale, rng))
        scales = []
        scale = max(1, min_scale)
        while scale <= max_scale:
            scales.append(scale)
            scale *= 2
        return scales or [max(height, width)]
    if isinstance(scales_spec, (int, float)):
        return [int(scales_spec)]
    if isinstance(scales_spec, list):
        return [int(s) for s in scales_spec if int(s) > 0]
    raise ValueError(f"Unsupported scales spec: {scales_spec}")


def modulate_with_noise(
    mean_value: np.ndarray,
    noise: np.ndarray,
    min_factor: float,
    max_factor: float,
    normalize_to_mean: bool,
) -> np.ndarray:
    factors = min_factor + (max_factor - min_factor) * noise
    if normalize_to_mean:
        mean_factor = float(np.mean(factors))
        if mean_factor > 0:
            factors = factors / mean_factor
    if mean_value.ndim == 1:
        mean_value = mean_value.reshape(1, 1, -1)
    return mean_value * factors[..., None]


def modulate_with_noise_torch(
    mean_value: "torch.Tensor",
    noise: "torch.Tensor",
    min_factor: float,
    max_factor: float,
    normalize_to_mean: bool,
) -> "torch.Tensor":
    factors = min_factor + (max_factor - min_factor) * noise
    if normalize_to_mean:
        mean_factor = float(factors.mean().item())
        if mean_factor > 0:
            factors = factors / mean_factor
    if mean_value.ndim == 1:
        mean_value = mean_value.view(1, 1, -1)
    return mean_value * factors[..., None]


def apply_fog(
    rgb: np.ndarray, depth_m: np.ndarray, k_field: np.ndarray, ls_field: np.ndarray
) -> np.ndarray:
    t = np.exp(-k_field * depth_m)
    t = t[..., None]
    return rgb * t + ls_field * (1.0 - t)


def apply_fog_torch(
    rgb: "torch.Tensor",
    depth_m: "torch.Tensor",
    k_field,
    ls_field,
) -> "torch.Tensor":
    if not torch.is_tensor(k_field):
        k_field = torch.tensor(k_field, device=rgb.device, dtype=rgb.dtype)
    if not torch.is_tensor(ls_field):
        ls_field = torch.tensor(ls_field, device=rgb.device, dtype=rgb.dtype)
    t = torch.exp(-k_field * depth_m)
    if t.ndim in (2, 3):
        t = t.unsqueeze(-1)
    return rgb * t + ls_field * (1.0 - t)


def select_model(config: dict, rng: np.random.Generator) -> str:
    selection = config.get("selection")
    if selection is None:
        if "fog_model" in config:
            return config["fog_model"]
        return "uniform"
    mode = selection.get("mode", "fixed")
    if mode == "fixed":
        return selection.get("model", "uniform")
    if mode == "weighted":
        weights = selection.get("weights", {})
        if not weights:
            raise ValueError("selection.weights must be provided for weighted mode")
        names = list(weights.keys())
        probs = np.array([weights[name] for name in names], dtype=np.float32)
        probs = probs / probs.sum()
        return str(rng.choice(names, p=probs))
    raise ValueError(f"Unsupported selection mode: {mode}")


def resolve_model_config(model_name: str, models_cfg: dict) -> dict:
    base = DEFAULT_MODEL_CONFIGS.get(model_name, {})
    override = models_cfg.get(model_name, {})
    return deep_merge(base, override)


def uses_estimated_airlight(al_spec) -> bool:
    return al_spec is None or al_spec in AIRLIGHT_METHODS


def broadcast_k_field(k_field: Any, height: int, width: int) -> np.ndarray:
    """Return ``k_field`` as a ``(H, W)`` float32 map (broadcasting if scalar)."""
    arr = np.asarray(k_field, dtype=np.float32)
    if arr.ndim == 0:
        return np.broadcast_to(arr, (height, width)).astype(np.float32, copy=True)
    if arr.shape == (height, width):
        return arr.astype(np.float32, copy=False)
    raise ValueError(
        f"k_field must be scalar or shape ({height}, {width}); got {arr.shape}"
    )


def broadcast_ls_field(ls_field: Any, height: int, width: int) -> np.ndarray:
    """Return ``ls_field`` as a ``(H, W, 3)`` float32 map (broadcasting if needed)."""
    arr = np.asarray(ls_field, dtype=np.float32)
    if arr.shape == (3,):
        return np.broadcast_to(arr, (height, width, 3)).astype(np.float32, copy=True)
    if arr.shape == (1, 1, 3):
        return np.broadcast_to(arr, (height, width, 3)).astype(np.float32, copy=True)
    if arr.shape == (height, width, 3):
        return arr.astype(np.float32, copy=False)
    raise ValueError(
        f"ls_field must have shape (3,), (1, 1, 3), or "
        f"({height}, {width}, 3); got {arr.shape}"
    )


def apply_model(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    model_name: str,
    model_cfg: dict,
    rng: np.random.Generator,
    contrast_threshold_default: float,
    estimated_airlight: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    """Apply a fog model to ``rgb``.

    Returns:
        Tuple ``(foggy, k_mean, ls_base, k_map, ls_map)``:

        * ``foggy``: ``(H, W, 3)`` foggy RGB image.
        * ``k_mean``: scalar mean scattering coefficient (for filenames/logs).
        * ``ls_base``: ``(3,)`` base atmospheric light (for filenames/logs).
        * ``k_map``: ``(H, W)`` β-field actually used (broadcast for uniform).
        * ``ls_map``: ``(H, W, 3)`` L_s-field actually used (broadcast for uniform).
    """
    if model_name not in DEFAULT_MODEL_CONFIGS:
        raise ValueError(f"Unsupported fog model: {model_name}")
    visibility = float(sample_value(model_cfg.get("visibility_m"), rng))
    contrast_threshold = float(
        sample_value(
            model_cfg.get("contrast_threshold", contrast_threshold_default), rng
        )
    )
    k_mean = visibility_to_k(visibility, contrast_threshold)

    al_spec = model_cfg.get("atmospheric_light", "from_sky")
    if uses_estimated_airlight(al_spec):
        ls_base = normalize_atmospheric_light(estimated_airlight)
    else:
        sampled_al = sample_value(al_spec, rng)
        ls_base = normalize_atmospheric_light(np.asarray(sampled_al))

    height, width = depth_m.shape

    if model_name == "uniform":
        ls_field = ls_base.reshape(1, 1, 3)
        foggy = apply_fog(rgb, depth_m, k_mean, ls_field)
        k_map = broadcast_k_field(k_mean, height, width)
        ls_map = broadcast_ls_field(ls_base, height, width)
        return foggy, k_mean, ls_base, k_map, ls_map

    if model_name in ("heterogeneous_k", "heterogeneous_k_ls"):
        k_cfg = model_cfg.get("k_hetero", {})
        k_scales = resolve_scales(k_cfg, height, width, rng)
        k_noise = perlin_fbm(height, width, k_scales, rng)
        min_factor = float(sample_value(k_cfg.get("min_factor", 1.0), rng))
        max_factor = float(sample_value(k_cfg.get("max_factor", 1.0), rng))
        k_field = modulate_with_noise(
            np.array([k_mean], dtype=np.float32),
            k_noise,
            min_factor,
            max_factor,
            bool(k_cfg.get("normalize_to_mean", False)),
        )[..., 0]
    else:
        k_field = k_mean

    if model_name in ("heterogeneous_ls", "heterogeneous_k_ls"):
        ls_cfg = model_cfg.get("ls_hetero", {})
        ls_scales = resolve_scales(ls_cfg, height, width, rng)
        ls_noise = perlin_fbm(height, width, ls_scales, rng)
        min_factor = float(sample_value(ls_cfg.get("min_factor", 1.0), rng))
        max_factor = float(sample_value(ls_cfg.get("max_factor", 1.0), rng))
        ls_field = modulate_with_noise(
            ls_base,
            ls_noise,
            min_factor,
            max_factor,
            bool(ls_cfg.get("normalize_to_mean", False)),
        )
        ls_field = np.clip(ls_field, 0.0, 1.0)
    else:
        ls_field = ls_base.reshape(1, 1, 3)

    foggy = apply_fog(rgb, depth_m, k_field, ls_field)
    k_map = broadcast_k_field(k_field, height, width)
    ls_map = broadcast_ls_field(ls_field, height, width)
    return foggy, k_mean, ls_base, k_map, ls_map
