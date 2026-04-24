from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from euler_preprocess.common.device import configure_device, iter_batches, torch_generator_for_index
from euler_preprocess.common.intrinsics import extract_intrinsics, planar_to_radial_depth, planar_to_radial_depth_torch
from euler_preprocess.common.io import load_json
from euler_preprocess.common.logging import get_logger, progress_bar
from euler_preprocess.common.noise import perlin_fbm_torch
from euler_preprocess.common.normalize import (
    _is_chw,
    _to_numpy,
    normalize_depth,
    normalize_rgb,
    normalize_rgb_torch,
    normalize_sky_mask,
)
from euler_preprocess.common.output import LegacyOutputBackend, OutputSlotSpec
from euler_preprocess.common.sampling import format_value, sample_value
from euler_preprocess.common.transform import Transform
from euler_preprocess.fog.airlight_from_sky import AirlightFromSky
from euler_preprocess.fog.dcp_airlight import DCPAirlight
from euler_preprocess.fog.dcp_heuristic_airlight import DCPHeuristicAirlight
from euler_preprocess.fog.logging import log_config
from euler_preprocess.fog.models import (
    AIRLIGHT_METHODS,
    DEFAULT_CONTRAST_THRESHOLD,
    DEFAULT_MODEL_CONFIGS,
    apply_fog_torch,
    apply_model,
    estimate_airlight_torch,
    modulate_with_noise_torch,
    normalize_atmospheric_light_torch,
    resolve_model_config,
    resolve_scales,
    select_model,
    uses_estimated_airlight,
    visibility_to_k,
)
from euler_loading.loaders.cpu.generic import (
    write_map_2d as _write_map_2d,
    write_map_3d as _write_map_3d,
)


SCATTERING_COEFFICIENT_SLOT = "scattering_coefficient"
ATMOSPHERIC_LIGHT_SLOT = "atmospheric_light"

# Use the canonical euler-loading ``generic.map_2d`` / ``map_3d`` modality
# annotations.  Auxiliary outputs are written as ``.npy`` files in the
# layout the matching loader expects (``map_2d`` → ``(H, W)``,
# ``map_3d`` → ``(C, H, W)`` after the writer's transpose).
_SCATTERING_INDEX_OVERLAY: dict[str, Any] = {
    "name": "scattering_coefficient",
    "type": "map_2d",
    "euler_train": {"used_as": "target"},
    "euler_loading": {
        "loader": "generic",
        "function": "map_2d",
    },
    "meta": {},
}

_ATMOSPHERIC_LIGHT_INDEX_OVERLAY: dict[str, Any] = {
    "name": "atmospheric_light",
    "type": "map_3d",
    "euler_train": {"used_as": "target"},
    "euler_loading": {
        "loader": "generic",
        "function": "map_3d",
    },
    "meta": {},
}

try:
    import torch
except ImportError:
    torch = None


class FogTransform(Transform):
    """Generate foggy versions of images from RGB + depth + semantic segmentation samples.

    Accepts an iterable of sample dicts (compatible with euler-loading's
    MultiModalDataset). Each sample must contain at minimum:
        "rgb":                      np.ndarray (H, W, 3)
        "depth":                    np.ndarray (H, W), float in meters
        "semantic_segmentation":    np.ndarray (H, W), boolean sky mask
        "id":         str
        "intrinsics": dict -- hierarchical modality containing ``"intrinsics"``
                      key mapping to a (3, 3) camera intrinsics matrix *K*.
                      When present, planar (z-buffer) depth is converted to
                      radial (Euclidean) depth before fog is applied.

    Optional:
        "full_id":  str -- hierarchical id from euler-loading (e.g.
                    "/Scene02/30-deg-right/Camera_0/00000").  When present,
                    the parent segments are used as subdirectories under the
                    model output folder so the dataset structure is preserved.
    """

    REQUIRED_MODALITIES: ClassVar[set[str]] = {"rgb", "depth", "semantic_segmentation"}
    REQUIRED_HIERARCHICAL_MODALITIES: ClassVar[set[str]] = set()
    SOURCE_MODALITY: ClassVar[str] = "rgb"
    OUTPUT_SLOT: ClassVar[str] = "rgb"
    OUTPUT_SLOTS: ClassVar[tuple[str, ...]] = (
        "rgb",
        SCATTERING_COEFFICIENT_SLOT,
        ATMOSPHERIC_LIGHT_SLOT,
    )
    OUTPUT_SLOT_SPECS: ClassVar[dict[str, OutputSlotSpec]] = {
        SCATTERING_COEFFICIENT_SLOT: OutputSlotSpec(
            source_modality="rgb",
            writer=_write_map_2d,
            index_overlay=_SCATTERING_INDEX_OVERLAY,
            output_extension=".npy",
        ),
        ATMOSPHERIC_LIGHT_SLOT: OutputSlotSpec(
            source_modality="rgb",
            writer=_write_map_3d,
            index_overlay=_ATMOSPHERIC_LIGHT_INDEX_OVERLAY,
            output_extension=".npy",
        ),
    }

    def __init__(
        self,
        config_path: str,
        out_path: str,
        suffix: str = "",
        output_backend: Any | None = None,
        output_backends: dict[str, Any] | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        if output_backends is not None:
            if "rgb" not in output_backends:
                raise ValueError(
                    "output_backends must contain the primary 'rgb' slot"
                )
            self.output_backends = dict(output_backends)
            self.output_backend = self.output_backends["rgb"]
        else:
            self.output_backend = output_backend or LegacyOutputBackend(out_path)
            self.output_backends = {"rgb": self.output_backend}
        self.out_path = self.output_backend.root
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
        airlight_method = self.config.get("airlight")
        if airlight_method is None:
            raise ValueError(
                "Config must specify 'airlight' key. "
                f"Supported values: {AIRLIGHT_METHODS}"
            )
        if airlight_method not in AIRLIGHT_METHODS:
            raise ValueError(
                f"Unknown airlight method '{airlight_method}'. "
                f"Supported values: {AIRLIGHT_METHODS}"
            )
        self.airlight_method = airlight_method
        dcp_heuristic_cfg = self.config.get("dcp_heuristic", {})
        if not isinstance(dcp_heuristic_cfg, dict):
            raise ValueError("Config key 'dcp_heuristic' must be an object")
        dcp_heuristic_kwargs = {
            key: dcp_heuristic_cfg[key]
            for key in (
                "patch_size",
                "top_percent",
                "white_bias",
                "cool_bias",
                "cool_target",
            )
            if key in dcp_heuristic_cfg
        }
        self.airlight_estimator_torch = None
        if airlight_method == "from_sky":
            self.airlight_estimator = AirlightFromSky(sky_depth_threshold=0.0)
        elif airlight_method == "dcp":
            self.airlight_estimator = DCPAirlight()
            if torch is not None:
                from euler_preprocess.fog.dcp_airlight_torch import DCPAirlightTorch
                self.airlight_estimator_torch = DCPAirlightTorch()
        elif airlight_method == "dcp_heuristic":
            self.airlight_estimator = DCPHeuristicAirlight(**dcp_heuristic_kwargs)
            if torch is not None:
                from euler_preprocess.fog.dcp_heuristic_airlight_torch import (
                    DCPHeuristicAirlightTorch,
                )
                self.airlight_estimator_torch = DCPHeuristicAirlightTorch(
                    **dcp_heuristic_kwargs
                )

    def run(self, samples: Iterable[dict]) -> list[Path]:
        """Run the fog transform. Alias for :meth:`generate_fog`."""
        return self.generate_fog(samples)

    def generate_fog(self, samples: Iterable[dict]) -> list[Path]:
        """Generate fog on the given samples.

        Args:
            samples: Iterable of dicts, each containing "rgb", "depth",
                     "semantic_segmentation", and "id" keys.

        Returns:
            List of output file paths.
        """
        if self.use_gpu:
            return self._generate_fog_gpu(samples)
        return self._generate_fog_cpu(samples)

    def _configure_device(self) -> None:
        self.torch_device, self.use_gpu = configure_device(self.device)

    def _generate_fog_cpu(self, samples: Iterable[dict]) -> list[Path]:
        try:
            total = len(samples)  # type: ignore[arg-type]
        except TypeError:
            samples = list(samples)
            total = len(samples)
        saved_paths: list[Path] = []

        with progress_bar(total, "CPU", self.logger) as bar:
            for index, sample in enumerate(samples):
                rgb = normalize_rgb(sample["rgb"])

                depth = normalize_depth(
                    sample["depth"], rgb.shape[:2], self.resize_depth_flag
                )
                depth = depth * self.depth_scale
                depth = np.maximum(depth, 0.0)

                intrinsics = extract_intrinsics(sample)
                if intrinsics is not None:
                    depth = planar_to_radial_depth(depth, intrinsics)

                sky_mask = normalize_sky_mask(sample["semantic_segmentation"])
                estimated_airlight = self.airlight_estimator.estimate_airlight(
                    rgb, sky_mask, sample_id=sample.get("id")
                )

                if self.seed is not None:
                    rng = np.random.default_rng(
                        np.random.SeedSequence([self.seed, index])
                    )
                else:
                    rng = self.base_rng

                model_name = select_model(self.config, rng)
                model_cfg = resolve_model_config(model_name, self.models_cfg)
                foggy, beta, airlight, k_map, ls_map = apply_model(
                    rgb,
                    depth,
                    model_name,
                    model_cfg,
                    rng,
                    self.contrast_threshold_default,
                    estimated_airlight,
                )

                if self.output_backend.is_source_backed:
                    saved_paths.append(self.output_backend.write(sample, foggy))
                else:
                    output_path = self._build_output_path(
                        sample["id"], model_name, beta, airlight,
                        full_id=sample.get("full_id"),
                    )
                    saved_paths.append(
                        self.output_backend.write(
                            sample,
                            foggy,
                            default_path=output_path,
                        )
                    )
                    self._write_model_config(model_name, model_cfg, saved_paths)

                self._write_auxiliary(
                    sample,
                    k_map=k_map,
                    ls_map=ls_map,
                    sample_id=sample["id"],
                    model_name=model_name,
                    full_id=sample.get("full_id"),
                )

                if bar is not None:
                    bar.update(1)
        self._finalize_backends()
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
    ) -> tuple[
        "torch.Tensor", float, "torch.Tensor", "torch.Tensor", "torch.Tensor"
    ]:
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
        if uses_estimated_airlight(al_spec):
            ls_base = normalize_atmospheric_light_torch(estimated_airlight_t).squeeze(0)
        else:
            sampled_al = sample_value(al_spec, rng)
            ls_base = normalize_atmospheric_light_torch(
                torch.tensor(sampled_al, device=self.torch_device, dtype=torch.float32)
            ).squeeze(0)

        height = int(depth_t.shape[0])
        width = int(depth_t.shape[1])

        if model_name == "uniform":
            ls_field = ls_base.view(1, 1, 3)
            foggy = apply_fog_torch(rgb_t, depth_t, k_mean, ls_field)
            k_map = self._broadcast_k_map_torch(k_mean, height, width)
            ls_map = self._broadcast_ls_map_torch(ls_base, height, width)
            return foggy, k_mean, ls_base, k_map, ls_map

        if model_name in ("heterogeneous_k", "heterogeneous_k_ls"):
            k_cfg = model_cfg.get("k_hetero", {})
            k_scales = resolve_scales(k_cfg, height, width, rng)
            k_noise = perlin_fbm_torch(
                height,
                width,
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
            ls_scales = resolve_scales(ls_cfg, height, width, rng)
            ls_noise = perlin_fbm_torch(
                height,
                width,
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

        foggy = apply_fog_torch(rgb_t, depth_t, k_field, ls_field)
        k_map = self._broadcast_k_map_torch(k_field, height, width)
        ls_map = self._broadcast_ls_map_torch(ls_field, height, width)
        return foggy, k_mean, ls_base, k_map, ls_map

    def _broadcast_k_map_torch(
        self, k_field, height: int, width: int
    ) -> "torch.Tensor":
        if torch.is_tensor(k_field) and k_field.shape == (height, width):
            return k_field.to(dtype=torch.float32)
        return torch.full(
            (height, width),
            float(k_field if not torch.is_tensor(k_field) else k_field.item()),
            device=self.torch_device,
            dtype=torch.float32,
        )

    def _broadcast_ls_map_torch(
        self, ls_field, height: int, width: int
    ) -> "torch.Tensor":
        if torch.is_tensor(ls_field) and ls_field.shape == (height, width, 3):
            return ls_field.to(dtype=torch.float32)
        # ls_field is (3,) base or (1, 1, 3) view — expand to full size.
        if torch.is_tensor(ls_field):
            base = ls_field.reshape(3)
        else:
            base = torch.tensor(ls_field, device=self.torch_device, dtype=torch.float32)
        return base.view(1, 1, 3).expand(height, width, 3).to(dtype=torch.float32)

    def _generate_fog_gpu(self, samples: Iterable[dict]) -> list[Path]:
        if torch is None or self.torch_device is None:
            raise RuntimeError("Torch device not configured for GPU execution.")
        device = self.torch_device
        try:
            total = len(samples)  # type: ignore[arg-type]
        except TypeError:
            samples = list(samples)
            total = len(samples)
        saved_paths: list[Path] = []

        with progress_bar(total, "GPU", self.logger) as bar:
            for batch in iter_batches(enumerate(samples), self.gpu_batch_size):
                items: list[dict] = []
                for global_index, sample in batch:
                    rgb = _to_numpy(sample["rgb"])
                    if _is_chw(rgb):
                        rgb = np.transpose(rgb, (1, 2, 0))
                    depth = normalize_depth(
                        sample["depth"], rgb.shape[:2], self.resize_depth_flag
                    )
                    intrinsics = extract_intrinsics(sample)
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
                            "meta": sample.get("meta"),
                            "rgb": rgb,
                            "depth": depth,
                            "intrinsics": intrinsics,
                            "sky_mask": normalize_sky_mask(sample["semantic_segmentation"]),
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

                        # Planar -> radial depth using intrinsics
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
                        if uses_estimated_airlight(al_spec):
                            if self.airlight_method == "from_sky":
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
                            elif self.airlight_method == "dcp_heuristic":
                                assert self.airlight_estimator_torch is not None
                                al_list = []
                                for idx in range(len(uniform_items)):
                                    sky_mask_t = torch.from_numpy(
                                        uniform_items[idx]["sky_mask"]
                                    ).to(device=device, dtype=torch.bool)
                                    al_t = self.airlight_estimator_torch.estimate_airlight(
                                        rgb_batch[idx],
                                        sky_mask_t,
                                        sample_id=uniform_items[idx]["sample_id"],
                                    )
                                    al_list.append(al_t)
                                airlight = torch.stack(al_list, dim=0)
                            else:
                                # Plain DCP: per-sample on GPU
                                assert self.airlight_estimator_torch is not None
                                al_list = []
                                for idx in range(len(uniform_items)):
                                    al_t = self.airlight_estimator_torch.compute(
                                        rgb_batch[idx]
                                    )
                                    al_list.append(al_t)
                                airlight = torch.stack(al_list, dim=0)
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

                        height = int(rgb_batch.shape[1])
                        width = int(rgb_batch.shape[2])
                        for idx, item in enumerate(uniform_items):
                            foggy_img = (
                                torch.clamp(foggy[idx], 0.0, 1.0).cpu().numpy()
                            )
                            airlight_np = ls_base[idx].detach().cpu().numpy()
                            sample_ref = {
                                "id": item["sample_id"],
                                "full_id": item.get("full_id"),
                                "meta": item.get("meta"),
                            }
                            if self.output_backend.is_source_backed:
                                saved_paths.append(
                                    self.output_backend.write(sample_ref, foggy_img)
                                )
                            else:
                                output_path = self._build_output_path(
                                    item["sample_id"],
                                    item["model_name"],
                                    k_means[idx],
                                    airlight_np,
                                    full_id=item.get("full_id"),
                                )
                                saved_paths.append(
                                    self.output_backend.write(
                                        sample_ref,
                                        foggy_img,
                                        default_path=output_path,
                                    )
                                )
                                self._write_model_config(
                                    item["model_name"], item["model_cfg"], saved_paths
                                )

                            if (
                                SCATTERING_COEFFICIENT_SLOT in self.output_backends
                                or ATMOSPHERIC_LIGHT_SLOT in self.output_backends
                            ):
                                k_map_np = np.full(
                                    (height, width),
                                    float(k_means[idx]),
                                    dtype=np.float32,
                                )
                                ls_map_np = np.broadcast_to(
                                    airlight_np.astype(np.float32, copy=False),
                                    (height, width, 3),
                                ).copy()
                                self._write_auxiliary(
                                    sample_ref,
                                    k_map=k_map_np,
                                    ls_map=ls_map_np,
                                    sample_id=item["sample_id"],
                                    model_name=item["model_name"],
                                    full_id=item.get("full_id"),
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
                        if self.airlight_method == "from_sky":
                            sky_mask_t = (
                                torch.from_numpy(item["sky_mask"]).to(device).bool()
                            )
                            estimated_airlight = estimate_airlight_torch(
                                rgb_t, sky_mask_t, sample_id=item["sample_id"]
                            )
                        elif self.airlight_method == "dcp_heuristic":
                            assert self.airlight_estimator_torch is not None
                            sky_mask_t = (
                                torch.from_numpy(item["sky_mask"]).to(device).bool()
                            )
                            estimated_airlight = (
                                self.airlight_estimator_torch.estimate_airlight(
                                    rgb_t,
                                    sky_mask_t,
                                    sample_id=item["sample_id"],
                                )
                            )
                        else:
                            assert self.airlight_estimator_torch is not None
                            estimated_airlight = (
                                self.airlight_estimator_torch.compute(rgb_t)
                            )
                        torch_gen = torch_generator_for_index(self.torch_device, self.seed, self.base_rng, item["index"])
                        foggy_t, beta, airlight_t, k_map_t, ls_map_t = (
                            self._apply_model_torch(
                                rgb_t,
                                depth_t,
                                item["model_name"],
                                item["model_cfg"],
                                item["rng"],
                                estimated_airlight,
                                torch_gen,
                            )
                        )
                        foggy_img = torch.clamp(foggy_t, 0.0, 1.0).cpu().numpy()
                        airlight_np = airlight_t.detach().cpu().numpy()
                        sample_ref = {
                            "id": item["sample_id"],
                            "full_id": item.get("full_id"),
                            "meta": item.get("meta"),
                        }
                        if self.output_backend.is_source_backed:
                            saved_paths.append(
                                self.output_backend.write(sample_ref, foggy_img)
                            )
                        else:
                            output_path = self._build_output_path(
                                item["sample_id"],
                                item["model_name"],
                                beta,
                                airlight_np,
                                full_id=item.get("full_id"),
                            )
                            saved_paths.append(
                                self.output_backend.write(
                                    sample_ref,
                                    foggy_img,
                                    default_path=output_path,
                                )
                            )
                            self._write_model_config(
                                item["model_name"], item["model_cfg"], saved_paths
                            )

                        if (
                            SCATTERING_COEFFICIENT_SLOT in self.output_backends
                            or ATMOSPHERIC_LIGHT_SLOT in self.output_backends
                        ):
                            k_map_np = k_map_t.detach().cpu().numpy()
                            ls_map_np = ls_map_t.detach().cpu().numpy()
                            self._write_auxiliary(
                                sample_ref,
                                k_map=k_map_np,
                                ls_map=ls_map_np,
                                sample_id=item["sample_id"],
                                model_name=item["model_name"],
                                full_id=item.get("full_id"),
                            )

                if bar is not None:
                    bar.update(len(batch))

        self._finalize_backends()
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
        if self.output_backend.is_source_backed:
            return
        if model_name in self._written_configs:
            return
        target_dir = self.out_path / model_name
        config_path = target_dir / "config.json"

        enriched_config = {**model_cfg, "size": len(saved_paths)}
        self.output_backend.write_json(config_path, enriched_config)
        self._written_configs.add(model_name)

    def _write_auxiliary(
        self,
        sample: dict,
        *,
        k_map: np.ndarray,
        ls_map: np.ndarray,
        sample_id: str,
        model_name: str,
        full_id: str | None,
    ) -> None:
        """Write the per-pixel β / L_s maps to their slots, if active."""
        scattering_backend = self.output_backends.get(SCATTERING_COEFFICIENT_SLOT)
        if scattering_backend is not None:
            self._write_aux_to_backend(
                scattering_backend,
                sample,
                k_map,
                sample_id=sample_id,
                model_name=model_name,
                full_id=full_id,
            )
        airlight_backend = self.output_backends.get(ATMOSPHERIC_LIGHT_SLOT)
        if airlight_backend is not None:
            self._write_aux_to_backend(
                airlight_backend,
                sample,
                ls_map,
                sample_id=sample_id,
                model_name=model_name,
                full_id=full_id,
            )

    def _write_aux_to_backend(
        self,
        backend: Any,
        sample: dict,
        value: np.ndarray,
        *,
        sample_id: str,
        model_name: str,
        full_id: str | None,
    ) -> None:
        if backend.is_source_backed:
            backend.write(sample, value)
            return
        # Legacy disk fallback: mirror the RGB output's hierarchy but emit .npy.
        base = backend.root / model_name
        if full_id:
            parts = [p for p in full_id.split("/") if p]
            if len(parts) > 1:
                base = base.joinpath(*parts[:-1])
        target = base / f"{sample_id}.npy"
        backend.write(sample, value, default_path=target)

    def _finalize_backends(self) -> None:
        for backend in self.output_backends.values():
            backend.finalize()


# Backward compatibility alias
Foggify = FogTransform
