"""Tests for inferring per-sample vs. hierarchical modality roles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from ds_crawler import DatasetWriter
from euler_loading import Modality, MultiModalDataset
from euler_loading.loaders.cpu.generic import write_intrinsics
from euler_loading.loaders.cpu.generic_dense_depth import write_depth

from euler_preprocess.cli import _run_transform
from euler_preprocess.common.dataset import build_dataset
from euler_preprocess.common.intrinsics import extract_intrinsics, planar_to_radial_depth
from euler_preprocess.common.normalize import normalize_depth
from euler_preprocess.radial.transform import RadialTransform

FRAMES = ("Scene01/Camera_0/0000", "Scene01/Camera_0/0001", "Scene02/Camera_0/0000")


def _K(fx: float) -> np.ndarray:
    return np.array([[fx, 0.0, 3.0], [0.0, fx, 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def _write_depth(root: Path, full_ids=FRAMES) -> str:
    writer = DatasetWriter(
        root,
        name="synthetic_depth",
        type="depth",
        separator=None,
        meta={"radial_depth": False, "scale_to_meters": 1.0, "range": [0, 1000]},
        euler_loading={"loader": "generic_dense_depth", "function": "depth"},
    )
    for full_id in full_ids:
        path = writer.get_path(full_id, full_id.rsplit("/", 1)[-1] + ".npy")
        write_depth(str(path), np.full((4, 6), 25.0, dtype=np.float32))
    writer.save_index()
    return str(root)


def _write_intrinsics(root: Path, matrices: dict[str, np.ndarray]) -> str:
    writer = DatasetWriter(
        root,
        name="synthetic_intrinsics",
        type="intrinsics",
        separator=None,
        euler_loading={"loader": "generic", "function": "intrinsics"},
    )
    for full_id, K in matrices.items():
        write_intrinsics(str(writer.get_path(full_id, f"{K[0, 0]:.0f}.npy")), K)
    writer.save_index()
    return str(root)


def _per_sample_intrinsics(root: Path) -> str:
    return _write_intrinsics(root, {fid: _K(500.0 + i) for i, fid in enumerate(FRAMES)})


def _per_scene_intrinsics(root: Path) -> str:
    return _write_intrinsics(
        root, {"Scene01/intrinsics": _K(600.0), "Scene02/intrinsics": _K(700.0)}
    )


def _roles(dataset) -> tuple[set[str], set[str]]:
    return set(dataset.modality_paths()), set(dataset.hierarchical_modality_paths())


def _fx_by_full_id(dataset) -> dict[str, float]:
    return {
        dataset[i]["full_id"]: float(extract_intrinsics(dataset[i])[0, 0])
        for i in range(len(dataset))
    }


@pytest.mark.parametrize("key", ["modalities", "hierarchical_modalities"])
def test_per_sample_intrinsics_are_joined_per_sample(tmp_path, key):
    config = {"modalities": {"depth": _write_depth(tmp_path / "depth")}}
    config.setdefault(key, {})["intrinsics"] = _per_sample_intrinsics(
        tmp_path / "intrinsics"
    )

    dataset = build_dataset(
        config,
        RadialTransform.REQUIRED_MODALITIES,
        RadialTransform.REQUIRED_HIERARCHICAL_MODALITIES,
    )

    assert _roles(dataset) == ({"depth", "intrinsics"}, set())
    assert _fx_by_full_id(dataset) == {
        "/Scene01/Camera_0/0000": 500.0,
        "/Scene01/Camera_0/0001": 501.0,
        "/Scene02/Camera_0/0000": 502.0,
    }


@pytest.mark.parametrize("key", ["modalities", "hierarchical_modalities"])
def test_per_scene_intrinsics_are_matched_hierarchically(tmp_path, key):
    config = {"modalities": {"depth": _write_depth(tmp_path / "depth")}}
    config.setdefault(key, {})["intrinsics"] = _per_scene_intrinsics(
        tmp_path / "intrinsics"
    )

    dataset = build_dataset(
        config,
        RadialTransform.REQUIRED_MODALITIES,
        RadialTransform.REQUIRED_HIERARCHICAL_MODALITIES,
    )

    assert _roles(dataset) == ({"depth"}, {"intrinsics"})
    assert _fx_by_full_id(dataset) == {
        "/Scene01/Camera_0/0000": 600.0,
        "/Scene01/Camera_0/0001": 600.0,
        "/Scene02/Camera_0/0000": 700.0,
    }


def test_seed_without_required_modalities_is_the_sample_modality(tmp_path):
    # The calibration sits at the same level as the frames and is listed first.
    calibration = _write_intrinsics(
        tmp_path / "intrinsics", {"Scene01/Camera_0/calibration": _K(800.0)}
    )
    depth = _write_depth(tmp_path / "depth", FRAMES[:2])

    dataset = build_dataset(
        {"modalities": {"intrinsics": calibration, "depth": depth}}, set()
    )

    assert _roles(dataset) == ({"depth"}, {"intrinsics"})
    assert len(dataset) == 2
    assert set(dataset[0]["intrinsics"]) == {"calibration"}


def test_modality_neither_per_sample_nor_attached_raises(tmp_path):
    config = {
        "modalities": {
            "depth": _write_depth(tmp_path / "depth"),
            "intrinsics": _write_intrinsics(
                tmp_path / "intrinsics", {"Scene99/intrinsics": _K(600.0)}
            ),
        },
    }

    with pytest.raises(ValueError, match="neither joined per sample nor matched"):
        build_dataset(config, {"depth"})


def test_modality_listed_under_both_keys_raises(tmp_path):
    intrinsics = _per_scene_intrinsics(tmp_path / "intrinsics")
    config = {
        "modalities": {"depth": _write_depth(tmp_path / "depth"), "intrinsics": intrinsics},
        "hierarchical_modalities": {"intrinsics": intrinsics},
    }

    with pytest.raises(ValueError, match="both 'modalities' and 'hierarchical_modalities'"):
        build_dataset(config, {"depth"})


@pytest.mark.parametrize("layout", [_per_sample_intrinsics, _per_scene_intrinsics])
def test_radial_cli_infers_intrinsics_role(tmp_path, layout):
    (tmp_path / "radial.json").write_text("{}", encoding="utf-8")
    config_path = tmp_path / "dataset.json"
    config_path.write_text(
        json.dumps(
            {
                "transform_config_path": "radial.json",
                "output_path": str(tmp_path / "radial_depth"),
                "modalities": {
                    "depth": _write_depth(tmp_path / "depth"),
                    "intrinsics": layout(tmp_path / "intrinsics"),
                },
            }
        ),
        encoding="utf-8",
    )
    source = build_dataset(json.loads(config_path.read_text()), {"depth"})

    assert _run_transform(argparse.Namespace(config=str(config_path)), RadialTransform) == 0

    output = MultiModalDataset({"depth": Modality(str(tmp_path / "radial_depth"))})
    assert len(output) == len(source) == len(FRAMES)
    for i in range(len(source)):
        expected = planar_to_radial_depth(
            normalize_depth(source[i]["depth"]), extract_intrinsics(source[i])
        )
        assert output[i]["full_id"] == source[i]["full_id"]
        np.testing.assert_allclose(normalize_depth(output[i]["depth"]), expected, rtol=1e-6)
