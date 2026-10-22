"""Regression tests for sky-mask and estimated-airlight normalisation."""

import json

import numpy as np
import pytest
from ds_crawler import DatasetWriter
from euler_loading.loaders.cpu import real_drive_sim as rds
from euler_loading.loaders.cpu.generic_dense_depth import write_depth

from euler_preprocess.common.dataset import build_dataset
from euler_preprocess.common.logging import get_logger
from euler_preprocess.common.normalize import normalize_sky_mask
from euler_preprocess.fog.atmospheric_light import AtmosphericLightResolver
from euler_preprocess.fog.models import (
    apply_model,
    normalize_atmospheric_light,
    resolve_model_config,
)
from euler_preprocess.fog.transform import FogTransform
from euler_preprocess.sky_depth.transform import SkyDepthTransform

try:
    import torch
except ImportError:
    torch = None

SKY = 23
# Cityscapes-style label map: every pixel has a non-zero class ID.
LABELS = np.array(
    [
        [SKY, SKY, SKY, SKY],
        [SKY, 7, 7, 11],
        [7, 7, 11, 11],
        [7, 7, 11, 11],
    ],
    dtype=np.int64,
)
UNIFORM_CFG = {
    "visibility_m": 100.0,
    "atmospheric_light": "from_sky",
    "airlight_dampening": False,
}


class TestNormalizeSkyMask:
    def test_class_map_raises(self):
        with pytest.raises(ValueError, match="sky_mask loader") as excinfo:
            normalize_sky_mask(LABELS[None])
        assert "7, 11, 23" in str(excinfo.value)

    def test_rgb_class_map_raises(self):
        # VKITTI2 GPU layout (3, H, W); vegetation (90, 240, 0) shares sky's red.
        colours = np.zeros((3, 6, 6), dtype=np.int64)
        colours[:, :2] = np.array([90, 200, 255])[:, None, None]
        colours[:, 2:] = np.array([90, 240, 0])[:, None, None]
        with pytest.raises(ValueError, match="boolean sky mask"):
            normalize_sky_mask(colours)

    def test_boolean_and_binary_masks_pass(self):
        mask = LABELS == SKY
        np.testing.assert_array_equal(normalize_sky_mask(mask), mask)
        binary = mask.astype(np.uint8)[..., None]
        np.testing.assert_array_equal(normalize_sky_mask(binary), mask)

    @pytest.mark.skipif(torch is None, reason="torch not installed")
    def test_gpu_loader_mask_passes(self):
        mask = torch.from_numpy(LABELS == SKY)[None]
        np.testing.assert_array_equal(normalize_sky_mask(mask), LABELS == SKY)


def test_fog_transform_rejects_class_maps(tmp_path):
    cfg = {
        "airlight": "from_sky",
        "device": "cpu",
        "models": {"uniform": UNIFORM_CFG},
        "selection": {"mode": "fixed", "model": "uniform"},
    }
    path = tmp_path / "fog.json"
    path.write_text(json.dumps(cfg))
    transform = FogTransform(config_path=str(path), out_path=str(tmp_path / "out"))
    sample = {
        "rgb": np.full(LABELS.shape + (3,), 0.5, dtype=np.float32),
        "depth": np.full(LABELS.shape, 20.0, dtype=np.float32),
        "semantic_segmentation": LABELS,
        "id": "frame",
    }
    with pytest.raises(ValueError, match="sky_mask loader"):
        transform.run([sample])


# Real Drive Sim class IDs: 29 is sky, 31 the class a relabelled head declares.
RDS_CLASSES = np.array([[29, 29, 7], [31, 7, 7]], dtype=np.uint8)
FRAMES = ("Scene01/Camera_0/0000", "Scene01/Camera_0/0001")


def _rds_dataset_config(tmp_path, skyclass, segmentation_entry=None) -> dict:
    """Depth plus an RDS ``class_segmentation`` index whose head declares *skyclass*."""
    depth = DatasetWriter(
        tmp_path / "depth",
        name="depth",
        type="depth",
        separator=None,
        meta={"radial_depth": False, "scale_to_meters": 1.0, "range": [0, 1000]},
        euler_loading={"loader": "generic_dense_depth", "function": "depth"},
    )
    segmentation = DatasetWriter(
        tmp_path / "segmentation",
        name="segmentation",
        type="semantic_segmentation",
        separator=None,
        meta={"skyclass": skyclass},
        euler_loading={"loader": "real_drive_sim", "function": "class_segmentation"},
    )
    for full_id in FRAMES:
        name = full_id.rsplit("/", 1)[-1]
        write_depth(str(depth.get_path(full_id, f"{name}.npy")), np.full((2, 3), 5.0, dtype=np.float32))
        rds.write_class_segmentation(str(segmentation.get_path(full_id, f"{name}.png")), RDS_CLASSES)
    depth.save_index()
    segmentation.save_index()
    entry = str(tmp_path / "segmentation")
    if segmentation_entry is not None:
        entry = {"path": entry, **segmentation_entry}
    return {
        "modalities": {"depth": str(tmp_path / "depth"), "semantic_segmentation": entry}
    }


def _run_sky_depth(tmp_path, config) -> np.ndarray:
    dataset = build_dataset(
        config,
        SkyDepthTransform.REQUIRED_MODALITIES,
        None,
        SkyDepthTransform.MODALITY_FUNCTIONS,
    )
    cfg_path = tmp_path / "sky_depth.json"
    cfg_path.write_text(json.dumps({"sky_depth_value": 900.0, "sanity_check_samples": 0}))
    transform = SkyDepthTransform(str(cfg_path), str(tmp_path / "out"))
    return np.load(transform.run(dataset)[0])


class TestSkyMaskFromDatasetHead:
    """The transforms read class-segmentation indexes through euler-loading's ``sky_mask``."""

    def test_sky_comes_from_the_head_skyclass(self, tmp_path):
        config = _rds_dataset_config(tmp_path, [29, 0, 0])
        np.testing.assert_array_equal(
            _run_sky_depth(tmp_path, config), np.where(RDS_CLASSES == 29, 900.0, 5.0)
        )

    def test_relabelled_head_moves_the_sky(self, tmp_path):
        config = _rds_dataset_config(tmp_path, [31, 0, 0])
        np.testing.assert_array_equal(
            _run_sky_depth(tmp_path, config), np.where(RDS_CLASSES == 31, 900.0, 5.0)
        )

    def test_function_null_keeps_the_class_map_and_fails_loudly(self, tmp_path):
        config = _rds_dataset_config(tmp_path, [29, 0, 0], {"function": None})
        with pytest.raises(ValueError, match="boolean sky mask"):
            _run_sky_depth(tmp_path, config)


class TestEstimatedAirlightRange:
    # A DCP-heuristic estimate whose luminance rescale overshoots one channel.
    ESTIMATE = [0.9, 1.03, 0.98]
    CLIPPED = [0.9, 1.0, 0.98]

    def test_estimate_overshoot_is_clipped_not_rescaled(self):
        result = normalize_atmospheric_light(np.array(self.ESTIMATE), allow_8bit=False)
        np.testing.assert_allclose(result, self.CLIPPED, rtol=1e-6)

    def test_configured_8bit_colour_is_still_rescaled(self):
        result = normalize_atmospheric_light(np.array([200, 210, 220]))
        np.testing.assert_allclose(result, np.array([200, 210, 220]) / 255.0, rtol=1e-6)

    def test_cpu_model_keeps_overshooting_estimate(self):
        _foggy, _beta, ls_base, _k_map, _ls_map = apply_model(
            np.full((4, 4, 3), 0.5, dtype=np.float32),
            np.full((4, 4), 20.0, dtype=np.float32),
            "uniform",
            resolve_model_config("uniform", {"uniform": UNIFORM_CFG}),
            np.random.default_rng(0),
            0.05,
            np.array(self.ESTIMATE, dtype=np.float32),
        )
        np.testing.assert_allclose(ls_base, self.CLIPPED, rtol=1e-6)

    @pytest.mark.skipif(torch is None, reason="torch not installed")
    def test_torch_sample_keeps_overshooting_estimate(self):
        _k_mean, ls_base = AtmosphericLightResolver("from_sky").resolve_model_torch(
            model_cfg=resolve_model_config("uniform", {"uniform": UNIFORM_CFG}),
            rng=np.random.default_rng(0),
            contrast_threshold_default=0.05,
            estimated_airlight_t=torch.tensor(self.ESTIMATE),
            device=torch.device("cpu"),
        )
        torch.testing.assert_close(ls_base, torch.tensor(self.CLIPPED))

    @pytest.mark.skipif(torch is None, reason="torch not installed")
    def test_torch_batch_overshoot_leaves_other_samples_unchanged(self, monkeypatch):
        resolver = AtmosphericLightResolver("from_sky")
        estimates = torch.tensor([[0.5, 0.6, 0.7], self.ESTIMATE])
        monkeypatch.setattr(resolver, "_estimate_batch_torch", lambda *a, **k: estimates)
        model_cfg = resolve_model_config("uniform", {"uniform": UNIFORM_CFG})
        items = [
            {"model_cfg": model_cfg, "rng": np.random.default_rng(index)}
            for index in range(2)
        ]
        _k_means, ls_base = resolver.resolve_uniform_batch_torch(
            rgb_batch=torch.zeros(2, 4, 4, 3),
            items=items,
            device=torch.device("cpu"),
            contrast_threshold_default=0.05,
        )
        torch.testing.assert_close(
            ls_base, torch.tensor([[0.5, 0.6, 0.7], self.CLIPPED])
        )


class TestScenarioSelectionLog:
    def _log_messages(self, tmp_path, caplog, scenario_profiles):
        cfg = {
            "airlight": "from_sky",
            "device": "cpu",
            "models": {"uniform": UNIFORM_CFG},
            "selection": {"mode": "fixed", "model": "heterogeneous_k_ls"},
            "scenario_profiles": scenario_profiles,
        }
        path = tmp_path / "scenarios.json"
        path.write_text(json.dumps(cfg))
        logger = get_logger()
        logger.addHandler(caplog.handler)
        try:
            FogTransform(config_path=str(path), out_path=str(tmp_path / "out"))
        finally:
            logger.removeHandler(caplog.handler)
        return caplog.messages

    def test_profiles_with_models_do_not_log_fixed_model(self, tmp_path, caplog):
        messages = self._log_messages(
            tmp_path,
            caplog,
            [
                {"name": "haze", "weight": 0.7, "model": "uniform"},
                {"name": "clear", "weight": 0.3, "model": "uniform", "clear_weather": True},
            ],
        )
        assert any("2 scenario_profiles (scope=sample)" in m for m in messages)
        assert "Scenario haze: weight=0.7 model=uniform" in messages
        assert not any("Model fixed" in m for m in messages)

    def test_profile_without_model_defers_to_selection(self, tmp_path, caplog):
        messages = self._log_messages(
            tmp_path,
            caplog,
            [{"name": "default_model", "weight": 1.0}],
        )
        assert "Scenario default_model: weight=1.0 model=<selection>" in messages
        assert "Model fixed: heterogeneous_k_ls" in messages
