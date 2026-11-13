from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from euler_preprocess.common.intrinsics import (
    planar_to_radial_depth,
    planar_to_radial_depth_torch,
)
from euler_preprocess.common.io import load_json
from euler_preprocess.common.normalize import (
    _is_chw,
    _to_numpy,
    normalize_depth,
    normalize_rgb,
    normalize_rgb_torch,
    normalize_sky_mask,
)
from euler_preprocess.common.device import torch_generator_for_index
from euler_preprocess.fog.transform import FogTransform

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional
    torch = None


FogInferenceMode = Literal["cpu", "gpu"]


@dataclass(frozen=True)
class FogInferenceResult:
    """In-memory fog render output for experimental single-sample inference."""

    rgb: np.ndarray
    model_name: str
    beta: float
    airlight: np.ndarray
    k_map: np.ndarray
    ls_map: np.ndarray
    scenario_name: str | None = None


def render_fog_image(
    *,
    rgb: Any,
    depth: Any,
    semantic_segmentation: Any = None,
    intrinsics: Any = None,
    config_path: str | Path | None = None,
    config: Mapping[str, Any] | None = None,
    scenario_profile_name: str | None = None,
    mode: FogInferenceMode = "cpu",
    sample_id: str | None = None,
    sample_index: int = 0,
    sky_class: Sequence[int | float] | None = None,
    depth_radial: bool | None = None,
) -> np.ndarray:
    """Render and return only the RGB image for one in-memory sample.

    See :func:`render_fog_sample` for the arguments.
    """

    return render_fog_sample(
        rgb=rgb,
        depth=depth,
        semantic_segmentation=semantic_segmentation,
        intrinsics=intrinsics,
        config_path=config_path,
        config=config,
        scenario_profile_name=scenario_profile_name,
        mode=mode,
        sample_id=sample_id,
        sample_index=sample_index,
        sky_class=sky_class,
        depth_radial=depth_radial,
    ).rgb


def render_fog_sample(
    *,
    rgb: Any,
    depth: Any,
    semantic_segmentation: Any = None,
    intrinsics: Any = None,
    config_path: str | Path | None = None,
    config: Mapping[str, Any] | None = None,
    scenario_profile_name: str | None = None,
    mode: FogInferenceMode = "cpu",
    sample_id: str | None = None,
    sample_index: int = 0,
    sky_class: Sequence[int | float] | None = None,
    depth_radial: bool | None = None,
) -> FogInferenceResult:
    """Render fog/camera artifacts for a single in-memory sample.

    The call accepts the modalities used by :class:`FogTransform` without
    requiring a dataset reader or output backend, and treats them the same way
    a ``euler-preprocess fog`` run does:

    - ``semantic_segmentation`` (optional) may be either a boolean sky mask or
      an RGB semantic label map; RGB maps are converted using ``sky_class``,
      the map's sky colour as its dataset head declares it in ``skyclass``
      (e.g. ``[29, 0, 0]`` for Real Drive Sim). Without it no pixel is sky.
    - ``intrinsics`` (optional) is a ``(3, 3)`` matrix *K*, a mapping with an
      ``"intrinsics"`` key, or ``{"fx", "fy", "cx", "cy"}``. It places the
      camera-profile optics and the sky fog path, and converts planar depth
      to the distance along the ray as ``depth_radial`` decides.
    - ``depth_radial`` is what the depth's dataset head declares in
      ``radial_depth``: ``True`` keeps the depth unconverted even with
      intrinsics, ``False`` requires intrinsics (or
      ``"planar_depth_as_distance": true`` in the config), ``None`` converts
      when intrinsics are present. See :class:`FogTransform`.
    """

    if mode not in ("cpu", "gpu"):
        raise ValueError("mode must be 'cpu' or 'gpu'")
    if config_path is None and config is None:
        raise ValueError("Either config_path or config must be provided")

    base_config = _load_config(config_path=config_path, config=config)
    runtime_config = _config_for_mode(base_config, mode)

    with tempfile.TemporaryDirectory(prefix="euler-fog-inference-") as tmpdir:
        tmp_path = Path(tmpdir)
        resolved_config_path = tmp_path / "config.json"
        resolved_config_path.write_text(json.dumps(runtime_config), encoding="utf-8")
        transform = FogTransform(
            config_path=str(resolved_config_path),
            out_path=str(tmp_path / "out"),
            depth_radial=depth_radial,
        )
        return _render_with_transform(
            transform,
            rgb=rgb,
            depth=depth,
            semantic_segmentation=semantic_segmentation,
            intrinsics=intrinsics,
            scenario_profile_name=scenario_profile_name,
            mode=mode,
            sample_id=sample_id,
            sample_index=sample_index,
            sky_class=sky_class,
        )


def _load_config(
    *,
    config_path: str | Path | None,
    config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if config_path is not None:
        loaded = load_json(Path(config_path))
        if config is None:
            return dict(loaded)
        merged = dict(loaded)
        merged.update(dict(config))
        return merged
    return dict(config or {})


def _config_for_mode(config: dict[str, Any], mode: FogInferenceMode) -> dict[str, Any]:
    resolved = dict(config)
    if mode == "cpu":
        resolved["device"] = "cpu"
        return resolved

    configured_device = str(resolved.get("device", "")).strip().lower()
    if configured_device in ("", "cpu"):
        resolved["device"] = "gpu"
    return resolved


def _render_with_transform(
    transform: FogTransform,
    *,
    rgb: Any,
    depth: Any,
    semantic_segmentation: Any,
    intrinsics: Any,
    scenario_profile_name: str | None,
    mode: FogInferenceMode,
    sample_id: str | None,
    sample_index: int,
    sky_class: Sequence[int | float] | None,
) -> FogInferenceResult:
    sample_id = sample_id or "inference_sample"
    rgb_np = normalize_rgb(rgb)
    depth_np = normalize_depth(
        depth,
        rgb_np.shape[:2],
        transform.resize_depth_flag,
    )
    intrinsics_np = _normalize_intrinsics(intrinsics)
    # The transform's own rules: radial_depth decides whether the intrinsics
    # convert the depth, and a missing sky mask means no pixel is sky.
    radial_intrinsics = transform._radial_intrinsics(intrinsics_np, {"id": sample_id})
    if semantic_segmentation is None:
        sky_mask = transform._sky_mask({}, rgb_np.shape[:2])
    else:
        sky_mask = _normalize_semantic_sky_mask(
            semantic_segmentation,
            target_shape=rgb_np.shape[:2],
            sky_class=sky_class,
        )

    rng = transform._rng_for(sample_index)
    scenario = _select_scenario_profile(transform, scenario_profile_name)
    plan = transform._resolve_render_plan(
        rng,
        scenario=scenario,
        sample_scenario=scenario is None,
    )

    if mode == "gpu":
        return _render_gpu(
            transform,
            rgb_np=rgb_np,
            depth_np=depth_np,
            sky_mask=sky_mask,
            intrinsics_np=intrinsics_np,
            radial_intrinsics=radial_intrinsics,
            plan=plan,
            rng=rng,
            sample_id=sample_id,
            sample_index=sample_index,
        )
    return _render_cpu(
        transform,
        rgb_np=rgb_np,
        depth_np=depth_np,
        sky_mask=sky_mask,
        intrinsics_np=intrinsics_np,
        radial_intrinsics=radial_intrinsics,
        plan=plan,
        rng=rng,
        sample_id=sample_id,
    )


def _select_scenario_profile(
    transform: FogTransform,
    scenario_profile_name: str | None,
) -> dict[str, Any] | None:
    if scenario_profile_name is None:
        return None
    for profile in transform.scenario_profiles:
        if transform._scenario_name(profile) == scenario_profile_name:
            return profile
    known = ", ".join(
        name
        for name in (
            transform._scenario_name(profile)
            for profile in transform.scenario_profiles
        )
        if name is not None
    )
    raise ValueError(
        f"Unknown scenario_profile_name '{scenario_profile_name}'. "
        f"Known: {known or '<none>'}"
    )


def _render_cpu(
    transform: FogTransform,
    *,
    rgb_np: np.ndarray,
    depth_np: np.ndarray,
    sky_mask: np.ndarray,
    intrinsics_np: np.ndarray | None,
    radial_intrinsics: np.ndarray | None,
    plan,
    rng: np.random.Generator,
    sample_id: str,
) -> FogInferenceResult:
    depth_m = np.maximum(depth_np * transform.depth_scale, 0.0)
    if radial_intrinsics is not None:
        depth_m = planar_to_radial_depth(depth_m, radial_intrinsics)

    result = transform.pipeline.process_np(
        rgb=rgb_np,
        depth_m=depth_m,
        sky_mask=sky_mask,
        model_name=plan.model_name,
        model_cfg=plan.model_cfg,
        rng=rng,
        sample_id=sample_id,
        intrinsics=intrinsics_np,
        airlight_method=plan.airlight_method,
        capture_artifacts=plan.capture_artifacts,
        clear_weather=plan.clear_weather,
    )
    return FogInferenceResult(
        rgb=np.asarray(result.rgb, dtype=np.float32),
        model_name=plan.model_name,
        beta=float(result.beta),
        airlight=np.asarray(result.airlight, dtype=np.float32),
        k_map=np.asarray(result.k_map, dtype=np.float32),
        ls_map=np.asarray(result.ls_map, dtype=np.float32),
        scenario_name=plan.scenario_name,
    )


def _render_gpu(
    transform: FogTransform,
    *,
    rgb_np: np.ndarray,
    depth_np: np.ndarray,
    sky_mask: np.ndarray,
    intrinsics_np: np.ndarray | None,
    radial_intrinsics: np.ndarray | None,
    plan,
    rng: np.random.Generator,
    sample_id: str,
    sample_index: int,
) -> FogInferenceResult:
    if torch is None or transform.torch_device is None:
        raise RuntimeError("Torch device not configured for GPU inference")

    device = transform.torch_device
    rgb_t = normalize_rgb_torch(rgb_np, device)
    depth_t = torch.from_numpy(depth_np).to(device=device, dtype=torch.float32)
    depth_t = torch.clamp(depth_t * transform.depth_scale, min=0.0)
    if radial_intrinsics is not None:
        K_t = torch.from_numpy(radial_intrinsics).to(device=device, dtype=torch.float32)
        depth_t = planar_to_radial_depth_torch(depth_t, K_t)

    sky_mask_t = torch.from_numpy(sky_mask).to(device=device, dtype=torch.bool)
    torch_gen = torch_generator_for_index(
        transform.torch_device,
        transform.seed,
        transform.base_rng,
        sample_index,
    )
    foggy_t, beta, airlight_t, k_map_t, ls_map_t = transform._process_torch_pipeline(
        rgb_t,
        depth_t,
        sky_mask_t,
        plan.model_name,
        plan.model_cfg,
        rng,
        torch_gen,
        sample_id=sample_id,
        intrinsics=intrinsics_np,
        airlight_method=plan.airlight_method,
        capture_artifacts=plan.capture_artifacts,
        clear_weather=plan.clear_weather,
    )
    return FogInferenceResult(
        rgb=torch.clamp(foggy_t, 0.0, 1.0).detach().cpu().numpy().astype(np.float32),
        model_name=plan.model_name,
        beta=float(beta),
        airlight=airlight_t.detach().cpu().numpy().astype(np.float32),
        k_map=k_map_t.detach().cpu().numpy().astype(np.float32),
        ls_map=ls_map_t.detach().cpu().numpy().astype(np.float32),
        scenario_name=plan.scenario_name,
    )


def _normalize_intrinsics(intrinsics: Any) -> np.ndarray | None:
    if intrinsics is None:
        return None
    if isinstance(intrinsics, Mapping):
        if "intrinsics" in intrinsics:
            intrinsics = intrinsics["intrinsics"]
        elif all(key in intrinsics for key in ("fx", "fy", "cx", "cy")):
            return np.array(
                [
                    [float(intrinsics["fx"]), 0.0, float(intrinsics["cx"])],
                    [0.0, float(intrinsics["fy"]), float(intrinsics["cy"])],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
    arr = _to_numpy(intrinsics).astype(np.float32)
    if arr.shape != (3, 3):
        raise ValueError(f"intrinsics must have shape (3, 3), got {arr.shape}")
    return arr


def _normalize_semantic_sky_mask(
    semantic_segmentation: Any,
    *,
    target_shape: tuple[int, int],
    sky_class: Sequence[int | float] | None,
) -> np.ndarray:
    semantic = _to_numpy(semantic_segmentation)
    if _is_chw(semantic):
        semantic = np.transpose(semantic, (1, 2, 0))

    if semantic.ndim == 3 and semantic.shape[-1] >= 3 and sky_class is not None:
        sky_value = np.asarray(sky_class, dtype=semantic.dtype).reshape(1, 1, -1)
        mask = np.all(semantic[..., : sky_value.shape[-1]] == sky_value, axis=-1)
    else:
        mask = normalize_sky_mask(semantic)

    if mask.shape != target_shape:
        raise ValueError(
            f"semantic_segmentation shape {mask.shape} does not match "
            f"image shape {target_shape}"
        )
    return mask.astype(bool, copy=False)
