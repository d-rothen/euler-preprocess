from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from PIL import Image

import logging

from euler_fog.fog.airlight_from_sky import AirlightFromSky
from euler_fog.fog.foggify_logging import get_logger, log_config, progress_bar

_torch_logger = logging.getLogger("foggify")

try:
    import torch
except ImportError:
    torch = None

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


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_image(path: Path, rgb: np.ndarray) -> None:
    rgb = np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    img = Image.fromarray(rgb, mode="RGB")
    img.save(path)


def resize_depth(depth: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    height, width = target_shape
    if depth.shape == (height, width):
        return depth
    depth_img = Image.fromarray(depth.astype(np.float32), mode="F")
    depth_img = depth_img.resize((width, height), resample=Image.BILINEAR)
    return np.asarray(depth_img, dtype=np.float32)


def lerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    return a + t * (b - a)


def fade(t: np.ndarray) -> np.ndarray:
    return t * t * t * (t * (t * 6 - 15) + 10)


def perlin_noise(
    height: int, width: int, scale: float, rng: np.random.Generator
) -> np.ndarray:
    scale = float(scale)
    if scale <= 0:
        raise ValueError(f"Perlin scale must be > 0, got {scale}")
    grid_w = int(math.ceil(width / scale)) + 1
    grid_h = int(math.ceil(height / scale)) + 1

    angles = rng.uniform(0, 2 * math.pi, size=(grid_h, grid_w))
    grads = np.stack((np.cos(angles), np.sin(angles)), axis=-1)

    x = np.arange(width, dtype=np.float32) / scale
    y = np.arange(height, dtype=np.float32) / scale
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1
    xf = x - x0
    yf = y - y0

    x0g = x0[None, :]
    x1g = x1[None, :]
    y0g = y0[:, None]
    y1g = y1[:, None]
    xf_g = xf[None, :]
    yf_g = yf[:, None]
    u = fade(xf_g)
    v = fade(yf_g)

    g00 = grads[y0g, x0g]
    g10 = grads[y0g, x1g]
    g01 = grads[y1g, x0g]
    g11 = grads[y1g, x1g]

    d00 = g00[..., 0] * xf_g + g00[..., 1] * yf_g
    d10 = g10[..., 0] * (xf_g - 1) + g10[..., 1] * yf_g
    d01 = g01[..., 0] * xf_g + g01[..., 1] * (yf_g - 1)
    d11 = g11[..., 0] * (xf_g - 1) + g11[..., 1] * (yf_g - 1)

    nx0 = lerp(d00, d10, u)
    nx1 = lerp(d01, d11, u)
    nxy = lerp(nx0, nx1, v)

    min_val = float(nxy.min())
    max_val = float(nxy.max())
    if math.isclose(max_val, min_val):
        return np.zeros_like(nxy, dtype=np.float32)
    nxy = (nxy - min_val) / (max_val - min_val)
    return nxy.astype(np.float32)


def perlin_fbm(
    height: int, width: int, scales: list[int], rng: np.random.Generator
) -> np.ndarray:
    total = np.zeros((height, width), dtype=np.float32)
    weight_sum = 0.0
    for scale in scales:
        noise = perlin_noise(height, width, scale, rng)
        weight = math.log2(scale) ** 2 if scale > 1 else 1.0
        total += noise * weight
        weight_sum += weight
    if weight_sum <= 0:
        return np.zeros((height, width), dtype=np.float32)
    total /= weight_sum
    return np.clip(total, 0.0, 1.0)


def perlin_noise_torch(
    height: int,
    width: int,
    scale: float,
    rng: "torch.Generator",
    device: "torch.device",
    batch_size: int = 1,
):
    scale = float(scale)
    if scale <= 0:
        raise ValueError(f"Perlin scale must be > 0, got {scale}")
    grid_w = int(math.ceil(width / scale)) + 1
    grid_h = int(math.ceil(height / scale)) + 1

    angles = torch.rand(
        (batch_size, grid_h, grid_w), device=device, generator=rng, dtype=torch.float32
    )
    angles = angles * (2.0 * math.pi)
    grads = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    x = torch.arange(width, device=device, dtype=torch.float32) / scale
    y = torch.arange(height, device=device, dtype=torch.float32) / scale
    x0 = torch.floor(x).to(torch.int64)
    y0 = torch.floor(y).to(torch.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    xf = x - x0
    yf = y - y0

    x0g = x0[None, :]
    x1g = x1[None, :]
    y0g = y0[:, None]
    y1g = y1[:, None]
    xf_g = xf[None, :]
    yf_g = yf[:, None]
    u = fade(xf_g)
    v = fade(yf_g)

    g00 = grads[:, y0g, x0g]
    g10 = grads[:, y0g, x1g]
    g01 = grads[:, y1g, x0g]
    g11 = grads[:, y1g, x1g]

    d00 = g00[..., 0] * xf_g + g00[..., 1] * yf_g
    d10 = g10[..., 0] * (xf_g - 1) + g10[..., 1] * yf_g
    d01 = g01[..., 0] * xf_g + g01[..., 1] * (yf_g - 1)
    d11 = g11[..., 0] * (xf_g - 1) + g11[..., 1] * (yf_g - 1)

    nx0 = lerp(d00, d10, u)
    nx1 = lerp(d01, d11, u)
    nxy = lerp(nx0, nx1, v)

    min_val = nxy.amin(dim=(1, 2), keepdim=True)
    max_val = nxy.amax(dim=(1, 2), keepdim=True)
    denom = max_val - min_val
    nxy = (nxy - min_val) / denom.clamp_min(1e-8)
    nxy = torch.where(denom <= 1e-8, torch.zeros_like(nxy), nxy)

    if batch_size == 1:
        return nxy[0]
    return nxy


def perlin_fbm_torch(
    height: int,
    width: int,
    scales: list[int],
    rng: "torch.Generator",
    device: "torch.device",
    batch_size: int = 1,
):
    total = torch.zeros((batch_size, height, width), device=device, dtype=torch.float32)
    weight_sum = 0.0
    for scale in scales:
        noise = perlin_noise_torch(height, width, scale, rng, device, batch_size)
        if noise.ndim == 2:
            noise = noise.unsqueeze(0)
        weight = math.log2(scale) ** 2 if scale > 1 else 1.0
        total += noise * weight
        weight_sum += weight
    if weight_sum <= 0:
        total = torch.zeros(
            (batch_size, height, width), device=device, dtype=torch.float32
        )
    else:
        total = total / weight_sum
    total = torch.clamp(total, 0.0, 1.0)
    if batch_size == 1:
        return total[0]
    return total


def visibility_to_k(visibility_m: float, contrast_threshold: float) -> float:
    if visibility_m <= 0:
        raise ValueError(f"Visibility must be > 0, got {visibility_m}")
    return -math.log(contrast_threshold) / visibility_m


def sample_value(spec, rng: np.random.Generator):
    if isinstance(spec, (int, float)):
        return float(spec)
    if isinstance(spec, list):
        return [sample_value(item, rng) for item in spec]
    if isinstance(spec, dict):
        if "dist" not in spec:
            if "value" in spec:
                return sample_value(spec["value"], rng)
            return {k: sample_value(v, rng) for k, v in spec.items()}
        dist = spec["dist"]
        if dist == "constant":
            return sample_value(spec.get("value", 0.0), rng)
        if dist == "uniform":
            return float(rng.uniform(spec["min"], spec["max"]))
        if dist == "normal":
            val = float(rng.normal(spec["mean"], spec["std"]))
            if "min" in spec or "max" in spec:
                val = float(
                    np.clip(val, spec.get("min", -np.inf), spec.get("max", np.inf))
                )
            return val
        if dist == "lognormal":
            val = float(rng.lognormal(spec["mean"], spec["sigma"]))
            if "min" in spec or "max" in spec:
                val = float(
                    np.clip(val, spec.get("min", -np.inf), spec.get("max", np.inf))
                )
            return val
        if dist == "choice":
            values = spec["values"]
            weights = spec.get("weights")
            idx = int(rng.choice(len(values), p=weights))
            return sample_value(values[idx], rng)
        raise ValueError(f"Unsupported dist: {dist}")
    if isinstance(spec, str):
        return spec
    raise ValueError(f"Unsupported spec type: {type(spec)}")


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
        _torch_logger.warning(
            "No sky pixels in segmentation mask%s; "
            "using default airlight fallback [1.0, 1.0, 1.0]",
            id_str,
        )
        return torch.ones(3, device=image.device, dtype=image.dtype)
    airlight_pixels = image[sky_mask]
    airlight = airlight_pixels.mean(dim=0)
    if not torch.all(torch.isfinite(airlight)):
        id_str = f" (sample {sample_id})" if sample_id else ""
        _torch_logger.warning(
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


def planar_to_radial_depth(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Convert planar (z-buffer) depth to radial (Euclidean) depth.

    For each pixel ``(u, v)`` the radial distance equals
    ``depth[v, u] * sqrt(((u - cx)/fx)**2 + ((v - cy)/fy)**2 + 1)``.

    Args:
        depth: ``(H, W)`` float32 planar depth in metres.
        K: ``(3, 3)`` float32 camera intrinsics matrix.

    Returns:
        ``(H, W)`` float32 radial depth in metres.
    """
    H, W = depth.shape
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    u_grid, v_grid = np.meshgrid(u, v)
    factor = np.sqrt(((u_grid - cx) / fx) ** 2 + ((v_grid - cy) / fy) ** 2 + 1.0)
    return depth * factor


def planar_to_radial_depth_torch(
    depth: "torch.Tensor", K: "torch.Tensor",
) -> "torch.Tensor":
    """Torch version of :func:`planar_to_radial_depth`.

    Args:
        depth: ``(H, W)`` or ``(B, H, W)`` float32 planar depth.
        K: ``(3, 3)`` float32 intrinsics matrix (single, shared across batch).

    Returns:
        Same shape as *depth*, converted to radial depth.
    """
    if depth.ndim == 3:
        H, W = depth.shape[1], depth.shape[2]
    else:
        H, W = depth.shape
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    u = torch.arange(W, device=depth.device, dtype=depth.dtype)
    v = torch.arange(H, device=depth.device, dtype=depth.dtype)
    v_grid, u_grid = torch.meshgrid(v, u, indexing="ij")
    factor = torch.sqrt(((u_grid - cx) / fx) ** 2 + ((v_grid - cy) / fy) ** 2 + 1.0)
    return depth * factor


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


def deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_model_config(model_name: str, models_cfg: dict) -> dict:
    base = DEFAULT_MODEL_CONFIGS.get(model_name, {})
    override = models_cfg.get(model_name, {})
    return deep_merge(base, override)


def apply_model(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    model_name: str,
    model_cfg: dict,
    rng: np.random.Generator,
    contrast_threshold_default: float,
    estimated_airlight: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray]:
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
    if al_spec == "from_sky" or al_spec is None:
        ls_base = normalize_atmospheric_light(estimated_airlight)
    else:
        sampled_al = sample_value(al_spec, rng)
        ls_base = normalize_atmospheric_light(np.asarray(sampled_al))

    if model_name == "uniform":
        ls_field = ls_base.reshape(1, 1, 3)
        return apply_fog(rgb, depth_m, k_mean, ls_field), k_mean, ls_base

    if model_name in ("heterogeneous_k", "heterogeneous_k_ls"):
        k_cfg = model_cfg.get("k_hetero", {})
        k_scales = resolve_scales(k_cfg, depth_m.shape[0], depth_m.shape[1], rng)
        k_noise = perlin_fbm(depth_m.shape[0], depth_m.shape[1], k_scales, rng)
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
        ls_scales = resolve_scales(ls_cfg, depth_m.shape[0], depth_m.shape[1], rng)
        ls_noise = perlin_fbm(depth_m.shape[0], depth_m.shape[1], ls_scales, rng)
        min_factor = float(sample_value(ls_cfg.get("min_factor", 1.0), rng))
        max_factor = float(sample_value(ls_cfg.get("max_factor", 1.0), rng))
        # Use the estimated airlight as ls_base when noising
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

    return apply_fog(rgb, depth_m, k_field, ls_field), k_mean, ls_base


def _is_chw(arr: np.ndarray) -> bool:
    """Heuristic: 3-D array is CHW when first dim is small and spatial dims are large."""
    return (
        arr.ndim == 3
        and arr.shape[0] in (1, 3, 4)
        and arr.shape[1] > 4
        and arr.shape[2] > 4
    )


def _to_numpy(data) -> np.ndarray:
    """Convert *data* (numpy, torch tensor, or PIL Image) to a numpy array."""
    if torch is not None and torch.is_tensor(data):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def normalize_rgb(image) -> np.ndarray:
    image = _to_numpy(image)
    if _is_chw(image):
        image = np.transpose(image, (1, 2, 0))
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    elif image.ndim == 3 and image.shape[-1] == 4:
        image = image[..., :3]
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0
    return np.clip(image, 0.0, 1.0)


def normalize_rgb_torch(image, device: "torch.device") -> "torch.Tensor":
    if torch.is_tensor(image):
        image_t = image.to(device=device, dtype=torch.float32)
        # CHW → HWC
        if image_t.ndim == 3 and image_t.shape[0] in (1, 3, 4) and image_t.shape[2] > 4:
            image_t = image_t.permute(1, 2, 0)
        is_int = not image_t.is_floating_point()
    else:
        image_np = np.asarray(image)
        is_int = np.issubdtype(image_np.dtype, np.integer)
        image_t = torch.from_numpy(image_np).to(device)
    if image_t.ndim == 2:
        image_t = image_t.unsqueeze(-1).repeat(1, 1, 3)
    elif image_t.ndim == 3 and image_t.shape[-1] == 4:
        image_t = image_t[..., :3]
    image_t = image_t.to(torch.float32)
    if is_int:
        image_t = image_t / 255.0
    else:
        if float(image_t.max()) > 1.0:
            image_t = image_t / 255.0
    return torch.clamp(image_t, 0.0, 1.0)


def normalize_depth(
    depth, target_shape: tuple[int, int], resize_depth_flag: bool
) -> np.ndarray:
    depth = _to_numpy(depth).astype(np.float32)
    # (1, H, W) → (H, W)  (GPU-loader channel-first format)
    if depth.ndim == 3 and depth.shape[0] == 1:
        depth = depth[0]
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if resize_depth_flag and depth.shape != target_shape:
        depth = resize_depth(depth, target_shape)
    if depth.shape != target_shape:
        raise ValueError(
            f"Depth shape {depth.shape} does not match image shape {target_shape}"
        )
    depth[~np.isfinite(depth)] = 0.0
    depth = np.maximum(depth, 0.0)
    return depth


def _extract_intrinsics(sample: dict) -> np.ndarray | None:
    """Extract the 3x3 intrinsics matrix from a sample dict.

    Hierarchical modalities are stored as ``sample[name][file_id]``.
    For intrinsics the convention is ``sample["intrinsics"]["intrinsics"]``.
    """
    intr_dict = sample.get("intrinsics")
    if intr_dict is None:
        return None
    raw = intr_dict.get("intrinsics") if isinstance(intr_dict, dict) else intr_dict
    if raw is None:
        return None
    return _to_numpy(raw).astype(np.float32)


def format_value(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text or "0"


class Foggify:
    """Generate foggy versions of images from RGB + depth + sky_mask samples.

    Accepts an iterable of sample dicts (compatible with euler-loading's
    MultiModalDataset). Each sample must contain at minimum:
        "rgb":        np.ndarray (H, W, 3)
        "depth":      np.ndarray (H, W), float in meters
        "sky_mask":   np.ndarray (H, W), boolean
        "id":         str
        "intrinsics": dict – hierarchical modality containing ``"intrinsics"``
                      key mapping to a (3, 3) camera intrinsics matrix *K*.
                      When present, planar (z-buffer) depth is converted to
                      radial (Euclidean) depth before fog is applied.

    Optional:
        "full_id":  str – hierarchical id from euler-loading (e.g.
                    "/Scene02/30-deg-right/Camera_0/00000").  When present,
                    the parent segments are used as subdirectories under the
                    model output folder so the dataset structure is preserved.
    """

    def __init__(
        self,
        config_path: str,
        out_path: str,
        suffix: str = "",
    ) -> None:
        self.config_path = Path(config_path)
        self.out_path = Path(out_path)
        self.suffix = suffix or ""

        self.config = load_json(self.config_path)
        self.models_cfg = (
            self.config.get("models") or self.config.get("fog_models") or {}
        )
        self.device = str(self.config.get("device", "cpu"))
        self.gpu_batch_size = max(1, int(self.config.get("gpu_batch_size", 4)))
        self.seed = self.config.get("seed")
        self.base_rng = np.random.default_rng(self.seed)
        self.contrast_threshold_default = float(
            sample_value(
                self.config.get("contrast_threshold", DEFAULT_CONTRAST_THRESHOLD),
                self.base_rng,
            )
        )
        self.depth_scale = float(self.config.get("depth_scale", 1.0))
        self.resize_depth_flag = bool(self.config.get("resize_depth", True))
        self._written_configs: set[str] = set()
        self.torch_device = None
        self.use_gpu = False
        self.logger = get_logger()
        self._configure_device()
        log_config(
            self.logger,
            self.config,
            str(self.config_path),
            str(self.out_path),
            self.device,
            self.use_gpu,
            torch is not None,
            str(self.torch_device) if self.torch_device else None,
            self.gpu_batch_size,
            0,
            self.depth_scale,
            self.resize_depth_flag,
            self.seed,
            self.contrast_threshold_default,
        )

        # Initialize the airlight estimator
        self.airlight_estimator = AirlightFromSky(sky_depth_threshold=0.0)

    def generate_fog(self, samples: Iterable[dict]) -> list[Path]:
        """Generate fog on the given samples.

        Args:
            samples: Iterable of dicts, each containing "rgb", "depth",
                     "sky_mask", and "id" keys.

        Returns:
            List of output file paths.
        """
        if self.use_gpu:
            return self._generate_fog_gpu(samples)
        return self._generate_fog_cpu(samples)

    def _configure_device(self) -> None:
        device = str(self.device).strip()
        device_key = device.lower()
        if device_key == "gpu":
            device = "cuda"
            device_key = "cuda"
        if device_key == "cpu":
            self.use_gpu = False
            self.torch_device = None
            return
        if torch is None:
            raise RuntimeError(
                f"Torch is required for device '{device}', but it is not installed."
            )
        torch_device = torch.device(device)
        if torch_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        self.torch_device = torch_device
        self.use_gpu = torch_device.type != "cpu"

    @staticmethod
    def _iter_batches(items: Iterable, batch_size: int):
        if batch_size <= 0:
            batch_size = 1
        batch: list = []
        for item in items:
            batch.append(item)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def _torch_generator_for_index(self, index: int) -> "torch.Generator":
        if self.torch_device is None:
            raise RuntimeError("Torch device not configured.")
        gen = torch.Generator(device=self.torch_device)
        if self.seed is not None:
            seed_seq = np.random.SeedSequence([self.seed, index])
            seed_val = int(seed_seq.generate_state(1, dtype=np.uint64)[0])
            seed_val = seed_val & 0x7FFFFFFFFFFFFFFF
        else:
            seed_val = int(self.base_rng.integers(0, np.iinfo(np.int64).max))
        gen.manual_seed(seed_val)
        return gen

    def _generate_fog_cpu(self, samples: Iterable[dict]) -> list[Path]:
        total = len(samples)  # type: ignore[arg-type]
        saved_paths: list[Path] = []

        with progress_bar(total, "CPU", self.logger) as bar:
            for index, sample in enumerate(samples):
                rgb = normalize_rgb(sample["rgb"])

                depth = normalize_depth(
                    sample["depth"], rgb.shape[:2], self.resize_depth_flag
                )
                depth = depth * self.depth_scale
                depth = np.maximum(depth, 0.0)

                intrinsics = _extract_intrinsics(sample)
                if intrinsics is not None:
                    depth = planar_to_radial_depth(depth, intrinsics)

                estimated_airlight = self.airlight_estimator.estimate_airlight(
                    rgb, sample["sky_mask"], sample_id=sample.get("id")
                )

                if self.seed is not None:
                    rng = np.random.default_rng(
                        np.random.SeedSequence([self.seed, index])
                    )
                else:
                    rng = self.base_rng

                model_name = select_model(self.config, rng)
                model_cfg = resolve_model_config(model_name, self.models_cfg)
                foggy, beta, airlight = apply_model(
                    rgb,
                    depth,
                    model_name,
                    model_cfg,
                    rng,
                    self.contrast_threshold_default,
                    estimated_airlight,
                )

                output_path = self._build_output_path(
                    sample["id"], model_name, beta, airlight,
                    full_id=sample.get("full_id"),
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                save_image(output_path, foggy)
                saved_paths.append(output_path)
                self._write_model_config(model_name, model_cfg, saved_paths)

                if bar is not None:
                    bar.update(1)
        return saved_paths

    def _apply_model_torch(
        self,
        rgb_t: "torch.Tensor",
        depth_t: "torch.Tensor",
        model_name: str,
        model_cfg: dict,
        rng: np.random.Generator,
        estimated_airlight_t: "torch.Tensor",
        torch_gen: "torch.Generator",
    ) -> tuple["torch.Tensor", float, "torch.Tensor"]:
        if model_name not in DEFAULT_MODEL_CONFIGS:
            raise ValueError(f"Unsupported fog model: {model_name}")
        visibility = float(sample_value(model_cfg.get("visibility_m"), rng))
        contrast_threshold = float(
            sample_value(
                model_cfg.get("contrast_threshold", self.contrast_threshold_default),
                rng,
            )
        )
        k_mean = visibility_to_k(visibility, contrast_threshold)

        al_spec = model_cfg.get("atmospheric_light", "from_sky")
        if al_spec == "from_sky" or al_spec is None:
            ls_base = normalize_atmospheric_light_torch(estimated_airlight_t).squeeze(0)
        else:
            sampled_al = sample_value(al_spec, rng)
            ls_base = normalize_atmospheric_light_torch(
                torch.tensor(sampled_al, device=self.torch_device, dtype=torch.float32)
            ).squeeze(0)

        if model_name == "uniform":
            ls_field = ls_base.view(1, 1, 3)
            return apply_fog_torch(rgb_t, depth_t, k_mean, ls_field), k_mean, ls_base

        if model_name in ("heterogeneous_k", "heterogeneous_k_ls"):
            k_cfg = model_cfg.get("k_hetero", {})
            k_scales = resolve_scales(k_cfg, depth_t.shape[0], depth_t.shape[1], rng)
            k_noise = perlin_fbm_torch(
                depth_t.shape[0],
                depth_t.shape[1],
                k_scales,
                torch_gen,
                self.torch_device,
            )
            min_factor = float(sample_value(k_cfg.get("min_factor", 1.0), rng))
            max_factor = float(sample_value(k_cfg.get("max_factor", 1.0), rng))
            k_field = modulate_with_noise_torch(
                torch.tensor([k_mean], device=self.torch_device, dtype=torch.float32),
                k_noise,
                min_factor,
                max_factor,
                bool(k_cfg.get("normalize_to_mean", False)),
            )[..., 0]
        else:
            k_field = k_mean

        if model_name in ("heterogeneous_ls", "heterogeneous_k_ls"):
            ls_cfg = model_cfg.get("ls_hetero", {})
            ls_scales = resolve_scales(ls_cfg, depth_t.shape[0], depth_t.shape[1], rng)
            ls_noise = perlin_fbm_torch(
                depth_t.shape[0],
                depth_t.shape[1],
                ls_scales,
                torch_gen,
                self.torch_device,
            )
            min_factor = float(sample_value(ls_cfg.get("min_factor", 1.0), rng))
            max_factor = float(sample_value(ls_cfg.get("max_factor", 1.0), rng))
            ls_field = modulate_with_noise_torch(
                ls_base,
                ls_noise,
                min_factor,
                max_factor,
                bool(ls_cfg.get("normalize_to_mean", False)),
            )
            ls_field = torch.clamp(ls_field, 0.0, 1.0)
        else:
            ls_field = ls_base.view(1, 1, 3)

        return apply_fog_torch(rgb_t, depth_t, k_field, ls_field), k_mean, ls_base

    def _generate_fog_gpu(self, samples: Iterable[dict]) -> list[Path]:
        if torch is None or self.torch_device is None:
            raise RuntimeError("Torch device not configured for GPU execution.")
        device = self.torch_device
        total = len(samples)  # type: ignore[arg-type]
        saved_paths: list[Path] = []

        with progress_bar(total, "GPU", self.logger) as bar:
            for batch in self._iter_batches(enumerate(samples), self.gpu_batch_size):
                items: list[dict] = []
                for global_index, sample in batch:
                    rgb = _to_numpy(sample["rgb"])
                    if _is_chw(rgb):
                        rgb = np.transpose(rgb, (1, 2, 0))
                    depth = normalize_depth(
                        sample["depth"], rgb.shape[:2], self.resize_depth_flag
                    )
                    intrinsics = _extract_intrinsics(sample)
                    if self.seed is not None:
                        rng = np.random.default_rng(
                            np.random.SeedSequence([self.seed, global_index])
                        )
                    else:
                        rng = self.base_rng
                    model_name = select_model(self.config, rng)
                    model_cfg = resolve_model_config(model_name, self.models_cfg)
                    items.append(
                        {
                            "sample_id": sample["id"],
                            "full_id": sample.get("full_id"),
                            "rgb": rgb,
                            "depth": depth,
                            "intrinsics": intrinsics,
                            "sky_mask": sample["sky_mask"],
                            "rng": rng,
                            "model_name": model_name,
                            "model_cfg": model_cfg,
                            "index": global_index,
                        }
                    )

                if not items:
                    continue

                grouped: dict[tuple[int, int], list[dict]] = {}
                for item in items:
                    shape = (item["rgb"].shape[0], item["rgb"].shape[1])
                    grouped.setdefault(shape, []).append(item)

                for group_items in grouped.values():
                    uniform_items = [
                        item
                        for item in group_items
                        if item["model_name"] == "uniform"
                    ]
                    other_items = [
                        item
                        for item in group_items
                        if item["model_name"] != "uniform"
                    ]

                    if uniform_items:
                        rgb_batch = torch.stack(
                            [
                                normalize_rgb_torch(item["rgb"], device)
                                for item in uniform_items
                            ],
                            dim=0,
                        )
                        depth_batch = torch.stack(
                            [
                                torch.from_numpy(item["depth"]).to(
                                    device=device, dtype=torch.float32
                                )
                                for item in uniform_items
                            ],
                            dim=0,
                        )
                        depth_batch = torch.clamp(
                            depth_batch * self.depth_scale, min=0.0
                        )

                        # Planar → radial depth using intrinsics
                        K_np = uniform_items[0].get("intrinsics")
                        if K_np is not None:
                            K_t = torch.from_numpy(K_np).to(
                                device=device, dtype=torch.float32,
                            )
                            depth_batch = planar_to_radial_depth_torch(
                                depth_batch, K_t,
                            )

                        # Resolve atmospheric_light per the model config
                        al_spec = uniform_items[0]["model_cfg"].get(
                            "atmospheric_light", "from_sky"
                        )
                        if al_spec == "from_sky" or al_spec is None:
                            sky_mask_batch = torch.stack(
                                [
                                    torch.from_numpy(item["sky_mask"]).to(device)
                                    for item in uniform_items
                                ],
                                dim=0,
                            ).to(torch.float32)
                            mask_sum = sky_mask_batch.sum(dim=(1, 2))
                            no_sky = mask_sum == 0
                            safe_sum = mask_sum.clone()
                            safe_sum[no_sky] = 1.0  # avoid division by zero
                            airlight = (
                                rgb_batch * sky_mask_batch[..., None]
                            ).sum(dim=(1, 2)) / safe_sum[:, None]
                            # Replace NaN rows (no sky) with white fallback
                            if no_sky.any():
                                for idx_ns in no_sky.nonzero(as_tuple=False):
                                    i = int(idx_ns.item())
                                    self.logger.warning(
                                        "No sky pixels in segmentation mask "
                                        "(sample %s); using default airlight "
                                        "fallback [1.0, 1.0, 1.0]",
                                        uniform_items[i]["sample_id"],
                                    )
                                airlight[no_sky] = 1.0
                            ls_base = normalize_atmospheric_light_torch(airlight)
                        else:
                            ls_values = []
                            for item in uniform_items:
                                sampled_al = sample_value(al_spec, item["rng"])
                                ls_values.append(
                                    normalize_atmospheric_light_torch(
                                        torch.tensor(
                                            sampled_al,
                                            device=device,
                                            dtype=torch.float32,
                                        )
                                    ).squeeze(0)
                                )
                            ls_base = torch.stack(ls_values, dim=0)

                        k_means: list[float] = []
                        for item in uniform_items:
                            visibility = float(
                                sample_value(
                                    item["model_cfg"].get("visibility_m"),
                                    item["rng"],
                                )
                            )
                            contrast_threshold = float(
                                sample_value(
                                    item["model_cfg"].get(
                                        "contrast_threshold",
                                        self.contrast_threshold_default,
                                    ),
                                    item["rng"],
                                )
                            )
                            k_means.append(
                                visibility_to_k(visibility, contrast_threshold)
                            )

                        k_tensor = torch.tensor(
                            k_means, device=device, dtype=rgb_batch.dtype
                        )
                        t = torch.exp(-depth_batch * k_tensor[:, None, None])
                        foggy = rgb_batch * t[..., None] + ls_base[
                            :, None, None, :
                        ] * (1.0 - t[..., None])

                        for idx, item in enumerate(uniform_items):
                            foggy_img = (
                                torch.clamp(foggy[idx], 0.0, 1.0).cpu().numpy()
                            )
                            airlight_np = ls_base[idx].detach().cpu().numpy()
                            output_path = self._build_output_path(
                                item["sample_id"],
                                item["model_name"],
                                k_means[idx],
                                airlight_np,
                                full_id=item.get("full_id"),
                            )
                            output_path.parent.mkdir(parents=True, exist_ok=True)
                            save_image(output_path, foggy_img)
                            saved_paths.append(output_path)
                            self._write_model_config(
                                item["model_name"], item["model_cfg"], saved_paths
                            )

                    for item in other_items:
                        rgb_t = normalize_rgb_torch(item["rgb"], device)
                        depth_t = torch.from_numpy(item["depth"]).to(
                            device=device, dtype=torch.float32
                        )
                        depth_t = torch.clamp(depth_t * self.depth_scale, min=0.0)
                        K_np = item.get("intrinsics")
                        if K_np is not None:
                            K_t = torch.from_numpy(K_np).to(
                                device=device, dtype=torch.float32,
                            )
                            depth_t = planar_to_radial_depth_torch(depth_t, K_t)
                        sky_mask_t = (
                            torch.from_numpy(item["sky_mask"]).to(device).bool()
                        )
                        estimated_airlight = estimate_airlight_torch(
                            rgb_t, sky_mask_t, sample_id=item["sample_id"]
                        )
                        torch_gen = self._torch_generator_for_index(item["index"])
                        foggy_t, beta, airlight_t = self._apply_model_torch(
                            rgb_t,
                            depth_t,
                            item["model_name"],
                            item["model_cfg"],
                            item["rng"],
                            estimated_airlight,
                            torch_gen,
                        )
                        foggy_img = torch.clamp(foggy_t, 0.0, 1.0).cpu().numpy()
                        airlight_np = airlight_t.detach().cpu().numpy()
                        output_path = self._build_output_path(
                            item["sample_id"],
                            item["model_name"],
                            beta,
                            airlight_np,
                            full_id=item.get("full_id"),
                        )
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        save_image(output_path, foggy_img)
                        saved_paths.append(output_path)
                        self._write_model_config(
                            item["model_name"], item["model_cfg"], saved_paths
                        )

                if bar is not None:
                    bar.update(len(batch))

        return saved_paths

    def _build_output_path(
        self,
        sample_id: str,
        model_name: str,
        beta: float,
        airlight: np.ndarray,
        full_id: str | None = None,
    ) -> Path:
        if self.suffix:
            filename = f"{sample_id}_{self.suffix}.png"
        else:
            beta_str = format_value(beta)
            r_str, g_str, b_str = (format_value(v) for v in airlight)
            filename = (
                f"beta_{beta_str}_airlight_{r_str}_{g_str}_{b_str}_rgb_{sample_id}.png"
            )
        base = self.out_path / model_name
        if full_id:
            # full_id is e.g. "/Scene02/30-deg-right/Camera_0/00000"
            # Use all segments except the last (the frame id) as subdirs.
            parts = [p for p in full_id.split("/") if p]
            if len(parts) > 1:
                base = base.joinpath(*parts[:-1])
        return base / filename

    def _write_model_config(
        self, model_name: str, model_cfg: dict, saved_paths: list
    ) -> None:
        if model_name in self._written_configs:
            return
        target_dir = self.out_path / model_name
        target_dir.mkdir(parents=True, exist_ok=True)
        config_path = target_dir / "config.json"

        enriched_config = {**model_cfg, "size": len(saved_paths)}

        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(enriched_config, handle, indent=2, sort_keys=True)
        self._written_configs.add(model_name)
