"""Fog without a sky mask, and depth handling driven by the head's ``radial_depth``."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ds_crawler import DatasetWriter
from euler_loading import Modality, MultiModalDataset
from euler_loading.loaders.cpu.generic_dense_depth import (
    write_depth,
    write_rgb,
    write_sky_mask,
)
from euler_loading.loaders.cpu.vkitti2 import write_intrinsics

from euler_preprocess.cli import _run_transform
from euler_preprocess.common.dataset import build_dataset, declared_radial_depth
from euler_preprocess.common.intrinsics import planar_to_radial_depth
from euler_preprocess.fog.airlight_from_sky import AirlightFromSky, NoSkyWarning
from euler_preprocess.fog.inference import render_fog_image, render_fog_sample
from euler_preprocess.fog.pipeline import FogProcessingPipeline
from euler_preprocess.fog.transform import FogTransform

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional
    torch = None

FOG_LOGGER = "euler-preprocess.fog"
FRAMES = ("Scene01/Camera_0/00000", "Scene01/Camera_0/00001", "Scene01/Camera_0/00002")
# Both sides above 4 px, so euler-preprocess recognises the GPU loaders' CHW layout.
H, W = 8, 10
# A short focal length, so the planar -> radial factor reaches ~1.5 at the corners.
K = np.array([[5.0, 0.0, 4.5], [0.0, 5.0, 3.5], [0.0, 0.0, 1.0]], dtype=np.float32)
UNIFORM = {
    "visibility_m": {"dist": "constant", "value": 60.0},
    "atmospheric_light": [0.7, 0.7, 0.7],
}


def _depth_map(index: int = 0) -> np.ndarray:
    return np.linspace(8.0, 30.0, H * W, dtype=np.float32).reshape(H, W) + index


def _rgb_map(index: int = 0) -> np.ndarray:
    base = np.linspace(20, 230, H * W, dtype=np.float32).reshape(H, W)
    return np.dstack([base, base[::-1], np.full((H, W), 90 + index, np.float32)]).astype(np.uint8)


# ---------------------------------------------------------------------------
# Synthetic ds-crawler datasets
# ---------------------------------------------------------------------------


def _write_rgb(root: Path) -> str:
    writer = DatasetWriter(
        root, name="rgb", type="rgb", separator=None, meta={"range": [0, 255]},
        euler_loading={"loader": "generic_dense_depth", "function": "rgb"},
    )
    for i, full_id in enumerate(FRAMES):
        write_rgb(str(writer.get_path(full_id, full_id.rsplit("/", 1)[-1] + ".png")), _rgb_map(i))
    writer.save_index()
    return str(root)


def _write_depth(root: Path, radial_depth: bool | None) -> str:
    """A depth index whose head declares *radial_depth*.

    The dataset contract requires ``radial_depth`` for the ``depth`` key, so
    an undeclared head (``None``) is written under a key without that rule.
    """
    meta = {"scale_to_meters": 1.0, "range": [0, 1000]}
    key = "map_2d"
    if radial_depth is not None:
        meta["radial_depth"] = radial_depth
        key = "depth"
    writer = DatasetWriter(
        root, name="depth", type=key, separator=None, meta=meta,
        euler_loading={"loader": "generic_dense_depth", "function": "depth"},
    )
    for i, full_id in enumerate(FRAMES):
        write_depth(str(writer.get_path(full_id, full_id.rsplit("/", 1)[-1] + ".npy")), _depth_map(i))
    writer.save_index()
    return str(root)


def _write_sky_masks(root: Path, sky_rows: int = 0) -> str:
    writer = DatasetWriter(
        root, name="sky", type="sky_mask", separator=None,
        meta={"sky_mask": [255, 255, 255]},
        euler_loading={"loader": "generic_dense_depth", "function": "sky_mask"},
    )
    mask = np.zeros((H, W), dtype=bool)
    mask[:sky_rows] = True
    for full_id in FRAMES:
        path = writer.get_path(full_id, full_id.rsplit("/", 1)[-1] + ".png")
        write_sky_mask(str(path), mask, {"sky_mask": [255, 255, 255]})
    writer.save_index()
    return str(root)


def _write_intrinsics(root: Path) -> str:
    writer = DatasetWriter(
        root, name="intrinsics", type="intrinsics", separator=None,
        euler_loading={"loader": "vkitti2", "function": "read_intrinsics"},
    )
    write_intrinsics(str(writer.get_path("Scene01/intrinsics", "intrinsics.txt")), K)
    writer.save_index()
    return str(root)


def _run_fog_cli(
    tmp_path: Path,
    name: str,
    *,
    radial_depth: bool | None,
    intrinsics: bool,
    sky_rows: int | None = None,
    fog_overrides: dict | None = None,
) -> list[np.ndarray]:
    """Run ``euler-preprocess fog`` and return the foggy images in sample order."""
    work = tmp_path / name
    work.mkdir()
    fog = {
        "airlight": "from_sky",
        "device": "cpu",
        "seed": 3,
        "models": {"uniform": UNIFORM},
        "selection": {"mode": "fixed", "model": "uniform"},
        **(fog_overrides or {}),
    }
    (work / "fog.json").write_text(json.dumps(fog), encoding="utf-8")
    modalities = {
        "rgb": _write_rgb(work / "rgb"),
        "depth": _write_depth(work / "depth", radial_depth),
    }
    if sky_rows is not None:
        modalities["semantic_segmentation"] = _write_sky_masks(work / "sky", sky_rows)
    if intrinsics:
        modalities["intrinsics"] = _write_intrinsics(work / "intrinsics")
    config_path = work / "dataset.json"
    config_path.write_text(
        json.dumps(
            {
                "transform_config_path": "fog.json",
                "output_path": str(work / "out"),
                "modalities": modalities,
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(config=str(config_path), strict=False)
    assert _run_transform(args, FogTransform) == 0
    output = MultiModalDataset({"rgb": Modality(str(work / "out"))})
    by_id = {output[i]["full_id"]: np.asarray(output[i]["rgb"]) for i in range(len(output))}
    return [by_id["/" + full_id] for full_id in FRAMES]


def _assert_same(a: list[np.ndarray], b: list[np.ndarray]) -> None:
    assert len(a) == len(b)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def _assert_differ(a: list[np.ndarray], b: list[np.ndarray]) -> None:
    assert len(a) == len(b)
    assert any(not np.array_equal(x, y) for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# 1. semantic_segmentation is optional
# ---------------------------------------------------------------------------


class TestOptionalSkyMask:
    def test_fog_cli_runs_without_semantic_segmentation(self, tmp_path):
        without = _run_fog_cli(tmp_path, "none", radial_depth=None, intrinsics=False)
        all_false = _run_fog_cli(tmp_path, "empty", radial_depth=None, intrinsics=False, sky_rows=0)
        _assert_same(without, all_false)

    def test_configured_sky_mask_is_still_used(self, tmp_path):
        fog = {"models": {"uniform": {**UNIFORM, "atmospheric_light": "from_sky"}}}
        without = _run_fog_cli(tmp_path, "none", radial_depth=None, intrinsics=False, fog_overrides=fog)
        with_sky = _run_fog_cli(
            tmp_path, "sky", radial_depth=None, intrinsics=False, sky_rows=1, fog_overrides=fog
        )
        _assert_differ(without, with_sky)

    def test_configured_sky_mask_anchors_roles_like_a_required_modality(self, tmp_path):
        config = {
            "modalities": {
                "rgb": _write_rgb(tmp_path / "rgb"),
                "depth": _write_depth(tmp_path / "depth", None),
                "semantic_segmentation": _write_sky_masks(tmp_path / "sky", 1),
            }
        }
        dataset = build_dataset(
            config,
            FogTransform.REQUIRED_MODALITIES,
            None,
            FogTransform.MODALITY_FUNCTIONS,
            optional_modalities=FogTransform.OPTIONAL_MODALITIES,
        )
        assert set(dataset.modality_paths()) == {"rgb", "depth", "semantic_segmentation"}
        assert np.asarray(dataset[0]["semantic_segmentation"]).dtype == bool

    def test_missing_mask_and_white_fallback_warn_once_per_run(self, tmp_path, caplog):
        config_path = tmp_path / "fog.json"
        config_path.write_text(
            json.dumps(
                {
                    "airlight": "from_sky",
                    "device": "cpu",
                    "seed": 1,
                    "augmentations": {"models": ["uniform"], "visibility_m": [20, 60]},
                }
            )
        )
        transform = FogTransform(config_path=str(config_path), out_path=str(tmp_path / "out"))
        transform.logger.addHandler(caplog.handler)
        samples = [
            {"rgb": _rgb_map(i), "depth": _depth_map(i), "id": f"f{i}"} for i in range(3)
        ]
        try:
            with caplog.at_level(logging.DEBUG, logger=FOG_LOGGER):
                paths = transform.run(samples)
        finally:
            transform.logger.removeHandler(caplog.handler)
        assert len(paths) == 6
        warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert sum("No 'semantic_segmentation'" in m for m in warnings) == 1
        assert sum("No sky pixels" in m for m in warnings) == 1
        assert transform.atmospheric_light.no_sky_warning.count == 6


class TestNoSkyWarning:
    def test_first_fallback_warns_later_ones_are_debug(self, caplog):
        estimator = AirlightFromSky(sky_depth_threshold=0.0)
        image = np.random.default_rng(0).random((8, 8, 3)).astype(np.float32)
        no_sky = np.zeros((8, 8), dtype=bool)
        with caplog.at_level(logging.DEBUG, logger=FOG_LOGGER):
            for i in range(4):
                estimator.estimate_airlight(image, no_sky, sample_id=f"s{i}")
        levels = [r.levelno for r in caplog.records if "No sky pixels" in r.getMessage()]
        assert levels == [logging.WARNING] + [logging.DEBUG] * 3
        assert "s0" in caplog.records[0].getMessage()
        assert estimator.no_sky_warning.count == 4

    @pytest.mark.skipif(torch is None, reason="torch not installed")
    def test_torch_estimate_reports_through_the_shared_warning(self, caplog):
        from euler_preprocess.fog.models import estimate_airlight_torch

        report = NoSkyWarning()
        image = torch.rand(8, 8, 3)
        no_sky = torch.zeros(8, 8, dtype=torch.bool)
        with caplog.at_level(logging.WARNING, logger=FOG_LOGGER):
            for i in range(3):
                result = estimate_airlight_torch(image, no_sky, sample_id=f"t{i}", no_sky_warning=report)
        assert torch.all(result == 1.0)
        assert report.count == 3
        assert sum("No sky pixels" in m for m in caplog.messages) == 1


# ---------------------------------------------------------------------------
# 2. depth handling follows the head's radial_depth
# ---------------------------------------------------------------------------


class TestRadialDepthFromHead:
    def test_radial_head_is_never_converted_even_with_intrinsics(self, tmp_path):
        radial_with_k = _run_fog_cli(tmp_path, "radial_k", radial_depth=True, intrinsics=True)
        radial = _run_fog_cli(tmp_path, "radial", radial_depth=True, intrinsics=False)
        legacy_with_k = _run_fog_cli(tmp_path, "legacy_k", radial_depth=None, intrinsics=True)
        _assert_same(radial_with_k, radial)
        _assert_differ(radial_with_k, legacy_with_k)

    def test_planar_head_is_converted_with_intrinsics(self, tmp_path):
        planar = _run_fog_cli(tmp_path, "planar_k", radial_depth=False, intrinsics=True)
        legacy = _run_fog_cli(tmp_path, "legacy_k", radial_depth=None, intrinsics=True)
        _assert_same(planar, legacy)

    def test_planar_head_without_intrinsics_fails(self, tmp_path):
        with pytest.raises(ValueError, match="radial_depth: false.*no intrinsics"):
            _run_fog_cli(tmp_path, "planar", radial_depth=False, intrinsics=False)

    def test_planar_depth_as_distance_opts_out_of_the_error(self, tmp_path):
        accepted = _run_fog_cli(
            tmp_path, "planar", radial_depth=False, intrinsics=False,
            fog_overrides={"planar_depth_as_distance": True},
        )
        legacy = _run_fog_cli(tmp_path, "legacy", radial_depth=None, intrinsics=False)
        _assert_same(accepted, legacy)

    def test_undeclared_head_keeps_the_legacy_behaviour(self, tmp_path):
        converted = _run_fog_cli(tmp_path, "legacy_k", radial_depth=None, intrinsics=True)
        as_is = _run_fog_cli(tmp_path, "legacy", radial_depth=None, intrinsics=False)
        _assert_differ(converted, as_is)

    @pytest.mark.parametrize("value", [True, False, None])
    def test_declared_radial_depth_reads_the_head(self, tmp_path, value):
        dataset = MultiModalDataset({"depth": Modality(_write_depth(tmp_path / "depth", value))})
        assert declared_radial_depth(dataset, "depth") == (value, "dataset head")

    def test_function_override_uses_the_loader_annotation(self):
        def distance(path, meta=None):  # pragma: no cover - never called
            raise AssertionError

        dataset = SimpleNamespace(
            get_modality_metadata=lambda name: {"radial_depth": False},
            get_loader=lambda name: distance,
            get_modality_index=lambda name: {
                "euler_loading": {"loader": "hypersim", "function": "depth"}
            },
            get_loader_meta=lambda name: {"meta": {"radial_depth": True}},
        )
        assert declared_radial_depth(dataset, "depth") == (
            True,
            "loader function 'distance'",
        )

    def test_planar_depth_as_distance_must_be_boolean(self, tmp_path):
        path = tmp_path / "fog.json"
        path.write_text(json.dumps({"airlight": "from_sky", "planar_depth_as_distance": "yes"}))
        with pytest.raises(ValueError, match="planar_depth_as_distance"):
            FogTransform(config_path=str(path), out_path=str(tmp_path / "out"))


def _transform(tmp_path: Path, name: str, depth_radial, *, config: dict | None = None) -> FogTransform:
    path = tmp_path / f"{name}.json"
    path.write_text(
        json.dumps(
            config
            or {
                "airlight": "from_sky",
                "device": "cpu",
                "seed": 5,
                "models": {"uniform": UNIFORM},
                "selection": {"mode": "fixed", "model": "uniform"},
            }
        )
    )
    return FogTransform(
        config_path=str(path), out_path=str(tmp_path / f"{name}_out"), depth_radial=depth_radial
    )


def _samples(with_intrinsics: bool) -> list[dict]:
    samples = []
    for i in range(2):
        sample = {"rgb": _rgb_map(i), "depth": _depth_map(i), "id": f"f{i}"}
        if with_intrinsics:
            sample["intrinsics"] = K
        samples.append(sample)
    return samples


def _depth_seen_by_renderer(transform: FogTransform, samples, monkeypatch) -> list[np.ndarray]:
    seen = []
    original = transform.pipeline.process_np

    def spy(**kwargs):
        seen.append(np.array(kwargs["depth_m"]))
        return original(**kwargs)

    monkeypatch.setattr(transform.pipeline, "process_np", spy)
    transform.run(samples)
    return seen


class TestRadialDepthInTheTransform:
    @pytest.mark.parametrize(
        ("depth_radial", "with_intrinsics", "converted"),
        [
            (True, True, False),
            (True, False, False),
            (False, True, True),
            (None, True, True),
            (None, False, False),
        ],
    )
    def test_cpu_distance(self, tmp_path, monkeypatch, depth_radial, with_intrinsics, converted):
        transform = _transform(tmp_path, "t", depth_radial)
        seen = _depth_seen_by_renderer(transform, _samples(with_intrinsics), monkeypatch)
        for i, depth in enumerate(seen):
            expected = planar_to_radial_depth(_depth_map(i), K) if converted else _depth_map(i)
            np.testing.assert_allclose(depth, expected, rtol=1e-6)

    def test_cpu_planar_without_intrinsics_raises(self, tmp_path):
        transform = _transform(tmp_path, "t", False)
        with pytest.raises(ValueError, match="'f0' has no intrinsics"):
            transform.run(_samples(False))


@pytest.mark.skipif(torch is None, reason="torch not installed")
class TestTorchPathways:
    """The torch pathways (run on the CPU device here) follow the same rules."""

    @staticmethod
    def _gpu(transform: FogTransform) -> FogTransform:
        transform.torch_device = torch.device("cpu")
        return transform

    @pytest.mark.parametrize("augmented", [False, True])
    def test_radial_head_matches_the_unconverted_render(self, tmp_path, augmented):
        config = {
            "airlight": "from_sky",
            "device": "cpu",
            "seed": 5,
            "models": {"uniform": UNIFORM},
            "selection": {"mode": "fixed", "model": "uniform"},
        }
        if augmented:
            config["augmentations"] = {"models": ["uniform"], "visibility_m": [30]}
        with_k = self._gpu(_transform(tmp_path, "with_k", True, config=config))
        without_k = self._gpu(_transform(tmp_path, "without_k", True, config=config))
        legacy_k = self._gpu(_transform(tmp_path, "legacy_k", None, config=config))
        out = [
            [np.asarray(_read(p)) for p in t._generate_fog_gpu(_samples(k))]
            for t, k in ((with_k, True), (without_k, False), (legacy_k, True))
        ]
        _assert_same(out[0], out[1])
        _assert_differ(out[0], out[2])

    def test_planar_head_without_intrinsics_raises(self, tmp_path):
        transform = self._gpu(_transform(tmp_path, "t", False))
        with pytest.raises(ValueError, match="no intrinsics"):
            transform._generate_fog_gpu(_samples(False))

    def test_missing_sky_mask_renders(self, tmp_path):
        transform = self._gpu(_transform(tmp_path, "t", None))
        assert len(transform._generate_fog_gpu(_samples(False))) == 2


def _read(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path))


# ---------------------------------------------------------------------------
# 4. Single-sample inference (render_fog_sample / render_fog_image, the API
#    behind preview renders) follows the same rules as a run
# ---------------------------------------------------------------------------

INFERENCE_CONFIG = {
    "airlight": "from_sky",
    "seed": 5,
    "models": {"uniform": UNIFORM},
    "selection": {"mode": "fixed", "model": "uniform"},
}


def _infer(**kwargs):
    return render_fog_sample(
        rgb=_rgb_map(), depth=_depth_map(), config=INFERENCE_CONFIG, mode="cpu", **kwargs
    )


@pytest.fixture
def depth_spy(monkeypatch) -> list[np.ndarray]:
    """The depths the CPU renderer receives."""
    seen: list[np.ndarray] = []
    original = FogProcessingPipeline.process_np

    def spy(self, **kwargs):
        seen.append(np.array(kwargs["depth_m"]))
        return original(self, **kwargs)

    monkeypatch.setattr(FogProcessingPipeline, "process_np", spy)
    return seen


class TestInference:
    @pytest.mark.parametrize(
        ("depth_radial", "with_intrinsics", "converted"),
        [
            (True, True, False),
            (True, False, False),
            (False, True, True),
            (None, True, True),
            (None, False, False),
        ],
    )
    def test_cpu_distance(self, depth_spy, depth_radial, with_intrinsics, converted):
        _infer(intrinsics=K if with_intrinsics else None, depth_radial=depth_radial)
        expected = planar_to_radial_depth(_depth_map(), K) if converted else _depth_map()
        assert len(depth_spy) == 1
        np.testing.assert_allclose(depth_spy[0], expected, rtol=1e-6)

    def test_radial_depth_with_intrinsics_renders_like_without(self):
        with_k = _infer(intrinsics=K, depth_radial=True)
        without_k = _infer(depth_radial=True)
        legacy = _infer(intrinsics=K)
        np.testing.assert_array_equal(with_k.rgb, without_k.rgb)
        assert not np.array_equal(with_k.rgb, legacy.rgb)

    def test_planar_depth_without_intrinsics_raises_unless_accepted(self):
        with pytest.raises(ValueError, match="'preview' has no intrinsics"):
            _infer(depth_radial=False, sample_id="preview")
        accepted = render_fog_sample(
            rgb=_rgb_map(),
            depth=_depth_map(),
            config={**INFERENCE_CONFIG, "planar_depth_as_distance": True},
            depth_radial=False,
        )
        np.testing.assert_array_equal(accepted.rgb, _infer().rgb)

    def test_depth_radial_must_be_a_bool_or_none(self):
        with pytest.raises(ValueError, match="depth_radial must be True, False or None"):
            _infer(depth_radial="yes")

    def test_missing_sky_mask_renders_like_an_empty_one(self, caplog):
        empty = _infer(semantic_segmentation=np.zeros((H, W), dtype=bool))
        with caplog.at_level(logging.WARNING, logger=FOG_LOGGER):
            without = _infer()
        np.testing.assert_array_equal(without.rgb, empty.rgb)
        assert any("No 'semantic_segmentation'" in r.getMessage() for r in caplog.records)

    def test_sky_mask_is_still_used(self):
        config = {
            **INFERENCE_CONFIG,
            "models": {"uniform": {**UNIFORM, "atmospheric_light": "from_sky"}},
        }
        sky = np.zeros((H, W), dtype=bool)
        sky[:2] = True
        kwargs = dict(rgb=_rgb_map(), depth=_depth_map(), config=config)
        with_sky = render_fog_image(semantic_segmentation=sky, **kwargs)
        without = render_fog_image(**kwargs)
        assert not np.array_equal(with_sky, without)

    @pytest.mark.skipif(torch is None, reason="torch not installed")
    def test_torch_pathway_follows_the_same_rules(self, tmp_path):
        # mode="gpu" needs an accelerator; run its torch pathway on the CPU
        # device, as TestTorchPathways does for the transform.
        from euler_preprocess.fog.inference import _render_with_transform

        def gpu(name, depth_radial, intrinsics):
            transform = _transform(tmp_path, name, depth_radial)
            transform.torch_device = torch.device("cpu")
            return _render_with_transform(
                transform,
                rgb=_rgb_map(),
                depth=_depth_map(),
                semantic_segmentation=None,
                intrinsics=intrinsics,
                scenario_profile_name=None,
                mode="gpu",
                sample_id="preview",
                sample_index=0,
                sky_class=None,
            ).rgb

        radial_with_k = gpu("a", True, K)
        np.testing.assert_allclose(radial_with_k, gpu("b", True, None), atol=1e-6)
        assert not np.allclose(radial_with_k, gpu("c", None, K), atol=1e-6)
        with pytest.raises(ValueError, match="'preview' has no intrinsics"):
            gpu("d", False, None)
