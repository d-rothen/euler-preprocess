"""Tests for FogTransform writing scattering-coefficient and airlight maps."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ds_crawler import DatasetWriter
from euler_loading import Modality, MultiModalDataset
from euler_loading.loaders.cpu.generic_dense_depth import (
    depth as load_depth,
    rgb as load_rgb,
    sky_mask as load_sky_mask,
    write_depth,
    write_rgb,
    write_sky_mask,
)

from euler_loading.loaders.cpu.generic import (
    map_2d as load_map_2d,
    map_3d as load_map_3d,
)

from euler_preprocess.common.output import prepare_output_backends
from euler_preprocess.fog.models import visibility_to_k
from euler_preprocess.fog.transform import (
    ATMOSPHERIC_LIGHT_SLOT,
    SCATTERING_COEFFICIENT_SLOT,
    FogTransform,
)


def _write_rgb_dataset(root: Path) -> None:
    writer = DatasetWriter(
        root,
        name="synthetic_rgb",
        type="rgb",
        euler_train={"used_as": "input", "modality_type": "rgb"},
        separator=None,
        meta={"range": [0, 255]},
        euler_loading={"loader": "generic_dense_depth", "function": "rgb"},
    )
    rgb = np.dstack([
        np.full((4, 6), 32, dtype=np.uint8),
        np.full((4, 6), 96, dtype=np.uint8),
        np.full((4, 6), 160, dtype=np.uint8),
    ])
    path = writer.get_path("Scene01/Camera_0/00001", "00001.png")
    write_rgb(str(path), rgb)
    writer.save_index()


def _write_depth_dataset(root: Path) -> None:
    writer = DatasetWriter(
        root,
        name="synthetic_depth",
        type="depth",
        euler_train={"used_as": "input", "modality_type": "depth"},
        separator=None,
        meta={
            "radial_depth": False,
            "scale_to_meters": 1.0,
            "range": [0, 1000],
        },
        euler_loading={"loader": "generic_dense_depth", "function": "depth"},
    )
    depth = np.full((4, 6), 25.0, dtype=np.float32)
    path = writer.get_path("Scene01/Camera_0/00001", "00001.npy")
    write_depth(str(path), depth)
    writer.save_index()


def _write_sky_mask_dataset(root: Path) -> None:
    writer = DatasetWriter(
        root,
        name="synthetic_sky_mask",
        type="sky_mask",
        euler_train={"used_as": "condition", "modality_type": "sky_mask"},
        separator=None,
        meta={"sky_mask": [255, 255, 255]},
        euler_loading={"loader": "generic_dense_depth", "function": "sky_mask"},
    )
    mask = np.zeros((4, 6), dtype=bool)
    mask[0, :] = True
    path = writer.get_path("Scene01/Camera_0/00001", "00001.png")
    write_sky_mask(str(path), mask, {"sky_mask": [255, 255, 255]})
    writer.save_index()


def _make_dataset(tmp_path: Path) -> MultiModalDataset:
    rgb_root = tmp_path / "rgb"
    depth_root = tmp_path / "depth"
    sky_root = tmp_path / "sky_mask"

    _write_rgb_dataset(rgb_root)
    _write_depth_dataset(depth_root)
    _write_sky_mask_dataset(sky_root)

    return MultiModalDataset(
        modalities={
            "rgb": Modality(str(rgb_root), loader=load_rgb, writer=write_rgb),
            "depth": Modality(str(depth_root), loader=load_depth, writer=write_depth),
            "semantic_segmentation": Modality(
                str(sky_root),
                loader=load_sky_mask,
                writer=write_sky_mask,
            ),
        },
    )


def _write_fog_config(path: Path, *, visibility_m: float = 200.0) -> Path:
    cfg = {
        "airlight": "from_sky",
        "device": "cpu",
        "seed": 7,
        "contrast_threshold": 0.05,
        "models": {
            "uniform": {
                "visibility_m": {"dist": "constant", "value": visibility_m},
                "atmospheric_light": [0.4, 0.5, 0.6],
            }
        },
        "selection": {"mode": "fixed", "model": "uniform"},
    }
    path.write_text(json.dumps(cfg))
    return path


def _build_pipeline_config(
    pipeline_root: Path,
    manifest_path: Path,
    *,
    include_scattering: bool,
    include_airlight: bool,
) -> dict:
    output_targets = [
        {
            "slot": "rgb",
            "datasetType": "rgb",
            "relativePath": "foggy_rgb",
            "path": str(pipeline_root / "foggy_rgb"),
            "storage": "directory",
        }
    ]
    if include_scattering:
        output_targets.append(
            {
                "slot": SCATTERING_COEFFICIENT_SLOT,
                "datasetType": "scattering_coefficient",
                "relativePath": "scattering",
                "path": str(pipeline_root / "scattering"),
                "storage": "directory",
            }
        )
    if include_airlight:
        output_targets.append(
            {
                "slot": ATMOSPHERIC_LIGHT_SLOT,
                "datasetType": "atmospheric_light",
                "relativePath": "airlight",
                "path": str(pipeline_root / "airlight"),
                "storage": "directory",
            }
        )
    return {
        "pipeline": {
            "output_root": str(pipeline_root),
            "outputs_manifest_path": str(manifest_path),
            "output_targets": output_targets,
        }
    }


def test_aux_slots_omitted_when_pipeline_has_no_targets(tmp_path: Path) -> None:
    """Without aux output_targets, prepare_output_backends only yields rgb."""
    dataset = _make_dataset(tmp_path)
    config = {"output_path": str(tmp_path / "foggy")}
    backends = prepare_output_backends(config, dataset, FogTransform)
    assert set(backends.keys()) == {"rgb"}


def test_writes_scattering_and_airlight_maps(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path)
    pipeline_root = tmp_path / "pipeline_root"
    manifest_path = pipeline_root / ".euler_pipeline" / "pipeline_outputs.json"
    config = _build_pipeline_config(
        pipeline_root,
        manifest_path,
        include_scattering=True,
        include_airlight=True,
    )

    visibility_m = 200.0
    backends = prepare_output_backends(config, dataset, FogTransform)
    assert set(backends.keys()) == {
        "rgb",
        SCATTERING_COEFFICIENT_SLOT,
        ATMOSPHERIC_LIGHT_SLOT,
    }

    transform = FogTransform(
        config_path=str(_write_fog_config(tmp_path / "fog_cfg.json", visibility_m=visibility_m)),
        out_path=str(backends["rgb"].root),
        output_backends=backends,
    )
    saved_paths = transform.run(dataset)

    rgb_path = pipeline_root / "foggy_rgb" / "Scene01" / "Camera_0" / "00001.png"
    assert rgb_path.exists()
    assert saved_paths == [rgb_path]

    scattering_path = (
        pipeline_root / "scattering" / "Scene01" / "Camera_0" / "00001.npy"
    )
    airlight_path = pipeline_root / "airlight" / "Scene01" / "Camera_0" / "00001.npy"
    assert scattering_path.exists()
    assert airlight_path.exists()

    # Round-trip via the canonical loaders so layout conventions match.
    k_map = load_map_2d(str(scattering_path))
    ls_map = load_map_3d(str(airlight_path))

    assert k_map.shape == (4, 6)
    assert k_map.dtype == np.float32
    expected_k = visibility_to_k(visibility_m, 0.05)
    np.testing.assert_allclose(k_map, expected_k, atol=1e-6)

    assert ls_map.shape == (4, 6, 3)
    assert ls_map.dtype == np.float32
    expected_ls = np.broadcast_to(
        np.array([0.4, 0.5, 0.6], dtype=np.float32), (4, 6, 3)
    )
    np.testing.assert_allclose(ls_map, expected_ls, atol=1e-6)

    # And verify the on-disk layout matches the writer's contract:
    # scattering = (H, W) directly, airlight = (C, H, W).
    raw_scattering = np.load(scattering_path)
    assert raw_scattering.shape == (4, 6)
    raw_airlight = np.load(airlight_path)
    assert raw_airlight.shape == (3, 4, 6)


def test_only_scattering_target_writes_only_scattering(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path)
    pipeline_root = tmp_path / "pipeline_root_scat_only"
    manifest_path = pipeline_root / ".euler_pipeline" / "pipeline_outputs.json"
    config = _build_pipeline_config(
        pipeline_root,
        manifest_path,
        include_scattering=True,
        include_airlight=False,
    )

    backends = prepare_output_backends(config, dataset, FogTransform)
    assert set(backends.keys()) == {"rgb", SCATTERING_COEFFICIENT_SLOT}

    transform = FogTransform(
        config_path=str(_write_fog_config(tmp_path / "fog_cfg.json")),
        out_path=str(backends["rgb"].root),
        output_backends=backends,
    )
    transform.run(dataset)

    assert (pipeline_root / "scattering" / "Scene01" / "Camera_0" / "00001.npy").exists()
    assert not (pipeline_root / "airlight").exists()


def test_pipeline_manifest_lists_all_active_slots(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path)
    pipeline_root = tmp_path / "pipeline_root_manifest"
    manifest_path = pipeline_root / ".euler_pipeline" / "pipeline_outputs.json"
    config = _build_pipeline_config(
        pipeline_root,
        manifest_path,
        include_scattering=True,
        include_airlight=True,
    )

    backends = prepare_output_backends(config, dataset, FogTransform)
    transform = FogTransform(
        config_path=str(_write_fog_config(tmp_path / "fog_cfg.json")),
        out_path=str(backends["rgb"].root),
        output_backends=backends,
    )
    transform.run(dataset)

    manifest = json.loads(manifest_path.read_text())
    assert manifest["version"] == 1
    slots = [target["slot"] for target in manifest["outputs"]]
    assert slots == ["rgb", SCATTERING_COEFFICIENT_SLOT, ATMOSPHERIC_LIGHT_SLOT]


def test_aux_outputs_carry_correct_index_metadata(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path)
    pipeline_root = tmp_path / "pipeline_root_meta"
    manifest_path = pipeline_root / ".euler_pipeline" / "pipeline_outputs.json"
    config = _build_pipeline_config(
        pipeline_root,
        manifest_path,
        include_scattering=True,
        include_airlight=True,
    )

    backends = prepare_output_backends(config, dataset, FogTransform)
    transform = FogTransform(
        config_path=str(_write_fog_config(tmp_path / "fog_cfg.json")),
        out_path=str(backends["rgb"].root),
        output_backends=backends,
    )
    transform.run(dataset)

    scattering_index = json.loads(
        (pipeline_root / "scattering" / ".ds_crawler" / "output.json").read_text()
    )
    assert scattering_index["name"] == "scattering_coefficient"
    assert scattering_index["type"] == "map_2d"
    assert scattering_index["euler_loading"]["loader"] == "generic"
    assert scattering_index["euler_loading"]["function"] == "map_2d"
    assert scattering_index["euler_train"]["used_as"] == "target"

    airlight_index = json.loads(
        (pipeline_root / "airlight" / ".ds_crawler" / "output.json").read_text()
    )
    assert airlight_index["name"] == "atmospheric_light"
    assert airlight_index["type"] == "map_3d"
    assert airlight_index["euler_loading"]["loader"] == "generic"
    assert airlight_index["euler_loading"]["function"] == "map_3d"
    assert airlight_index["euler_train"]["used_as"] == "target"


def test_primary_slot_auto_selected_when_aliased(tmp_path: Path) -> None:
    """A pipeline target whose slot is *not* one of the aux slot names is
    automatically picked up as the primary RGB target — even when its slot
    name doesn't match the transform's primary slot (e.g. ``"fog"``).
    """
    dataset = _make_dataset(tmp_path)
    pipeline_root = tmp_path / "pipeline_root_alias"
    manifest_path = pipeline_root / ".euler_pipeline" / "pipeline_outputs.json"
    config = {
        "pipeline": {
            "output_root": str(pipeline_root),
            "outputs_manifest_path": str(manifest_path),
            "output_targets": [
                {
                    "slot": "fog",
                    "datasetType": "rgb",
                    "relativePath": "foggy_rgb.zip",
                    "path": str(pipeline_root / "foggy_rgb.zip"),
                    "storage": "zip",
                },
                {
                    "slot": ATMOSPHERIC_LIGHT_SLOT,
                    "datasetType": "rgb",
                    "relativePath": "atmospheric_light.zip",
                    "path": str(pipeline_root / "atmospheric_light.zip"),
                    "storage": "zip",
                },
                {
                    "slot": SCATTERING_COEFFICIENT_SLOT,
                    "datasetType": "rgb",
                    "relativePath": "scattering_coefficient.zip",
                    "path": str(pipeline_root / "scattering_coefficient.zip"),
                    "storage": "zip",
                },
            ],
        }
    }

    backends = prepare_output_backends(config, dataset, FogTransform)

    assert set(backends.keys()) == {
        "rgb",
        SCATTERING_COEFFICIENT_SLOT,
        ATMOSPHERIC_LIGHT_SLOT,
    }
    # Primary backend points at the "fog" target, not a literal "rgb" target.
    assert backends["rgb"].root == pipeline_root / "foggy_rgb.zip"

    transform = FogTransform(
        config_path=str(_write_fog_config(tmp_path / "fog_cfg.json")),
        out_path=str(backends["rgb"].root),
        output_backends=backends,
    )
    transform.run(dataset)

    import zipfile

    with zipfile.ZipFile(pipeline_root / "foggy_rgb.zip", "r") as zf:
        assert "Scene01/Camera_0/00001.png" in zf.namelist()
    with zipfile.ZipFile(pipeline_root / "scattering_coefficient.zip", "r") as zf:
        assert "Scene01/Camera_0/00001.npy" in zf.namelist()
    with zipfile.ZipFile(pipeline_root / "atmospheric_light.zip", "r") as zf:
        assert "Scene01/Camera_0/00001.npy" in zf.namelist()

    # Manifest still lists every active slot in declaration order.
    manifest = json.loads(manifest_path.read_text())
    slots = [target["slot"] for target in manifest["outputs"]]
    assert slots == ["fog", SCATTERING_COEFFICIENT_SLOT, ATMOSPHERIC_LIGHT_SLOT]


def test_apply_model_returns_full_size_maps_for_uniform() -> None:
    """Sanity-check the broadcast logic on the model layer."""
    from euler_preprocess.fog.models import apply_model

    rng = np.random.default_rng(0)
    rgb = np.full((10, 12, 3), 0.5, dtype=np.float32)
    depth = np.full((10, 12), 30.0, dtype=np.float32)
    estimated = np.array([0.9, 0.85, 0.8], dtype=np.float32)
    cfg = {
        "visibility_m": {"dist": "constant", "value": 100.0},
        "atmospheric_light": "from_sky",
    }

    foggy, k_mean, ls_base, k_map, ls_map = apply_model(
        rgb,
        depth,
        "uniform",
        cfg,
        rng,
        contrast_threshold_default=0.05,
        estimated_airlight=estimated,
    )

    assert foggy.shape == (10, 12, 3)
    assert k_map.shape == (10, 12)
    assert ls_map.shape == (10, 12, 3)
    np.testing.assert_allclose(k_map, k_mean)
    np.testing.assert_allclose(ls_map, np.broadcast_to(ls_base, ls_map.shape))


def test_apply_model_returns_spatial_fields_for_heterogeneous() -> None:
    """Heterogeneous models should return the actual non-constant maps used."""
    from euler_preprocess.fog.models import apply_model

    rng = np.random.default_rng(123)
    rgb = np.full((16, 16, 3), 0.5, dtype=np.float32)
    depth = np.full((16, 16), 50.0, dtype=np.float32)
    estimated = np.array([0.8, 0.8, 0.9], dtype=np.float32)
    cfg = {
        "visibility_m": {"dist": "constant", "value": 80.0},
        "atmospheric_light": "from_sky",
        "k_hetero": {
            "scales": [4],
            "min_factor": 0.0,
            "max_factor": 1.0,
            "normalize_to_mean": False,
        },
    }

    _, k_mean, _, k_map, _ = apply_model(
        rgb,
        depth,
        "heterogeneous_k",
        cfg,
        rng,
        contrast_threshold_default=0.05,
        estimated_airlight=estimated,
    )

    assert k_map.shape == (16, 16)
    # Spatially varying — there should be actual variance in the field.
    assert float(k_map.std()) > 0.0
