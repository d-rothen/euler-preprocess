from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar

import numpy as np

from euler_preprocess.common.device import configure_device, iter_batches, torch_generator_for_index
from euler_preprocess.common.intrinsics import extract_intrinsics, planar_to_radial_depth, planar_to_radial_depth_torch
from euler_preprocess.common.io import load_json, save_image
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
    visibility_to_k,
)

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
        self.airlight_estimator_torch = None
        if airlight_method == "from_sky":
            self.airlight_estimator = AirlightFromSky(sky_depth_threshold=0.0)
        elif airlight_method == "dcp":
            self.airlight_estimator = DCPAirlight()
            if torch is not None:
                from euler_preprocess.fog.dcp_airlight_torch import DCPAirlightTorch
                self.airlight_estimator_torch = DCPAirlightTorch()
        elif airlight_method == "dcp_heuristic":
            self.airlight_estimator = DCPHeuristicAirlight()
            if torch is not None:
                from euler_preprocess.fog.dcp_heuristic_airlight_torch import (
                    DCPHeuristicAirlightTorch,
                )
                self.airlight_estimator_torch = DCPHeuristicAirlightTorch()

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
                        if al_spec == "from_sky" or al_spec is None:
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
                            else:
                                # DCP / DCP heuristic: per-sample on GPU
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
                        if self.airlight_method == "from_sky":
                            sky_mask_t = (
                                torch.from_numpy(item["sky_mask"]).to(device).bool()
                            )
                            estimated_airlight = estimate_airlight_torch(
                                rgb_t, sky_mask_t, sample_id=item["sample_id"]
                            )
                        else:
                            assert self.airlight_estimator_torch is not None
                            estimated_airlight = (
                                self.airlight_estimator_torch.compute(rgb_t)
                            )
                        torch_gen = torch_generator_for_index(self.torch_device, self.seed, self.base_rng, item["index"])
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


# Backward compatibility alias
Foggify = FogTransform
