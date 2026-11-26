"""Transforms from installed packages: the ``euler.transforms`` entry-point group."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pytest
from ds_crawler import DatasetWriter
from euler_dataset_contract.extensions import ExtensionError
from euler_loading import Modality, MultiModalDataset
from euler_loading.loaders.cpu.generic_dense_depth import write_depth

from euler_preprocess.capabilities import transform_capability_report
from euler_preprocess.cli import build_parser, main, run_transform
from euler_preprocess.common.normalize import normalize_depth
from euler_preprocess.fog.transform import FogTransform
from euler_preprocess.registry import (
    BUILTIN_TRANSFORMS,
    get_transform,
    run_options,
    transform_catalog,
    transform_problems,
)

from .extension_fakes import install, plugin
from .plugin_transforms import ScaleDepthTransform

FRAMES = ("Scene01/Camera_0/0000", "Scene01/Camera_0/0001")
SCALE = "tests.plugin_transforms:ScaleDepthTransform"


def _problems() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for problem in transform_problems():
        found.setdefault(problem.name, []).append(problem.code)
    return found


def _names() -> list[str]:
    return list(transform_catalog().entries)


@pytest.fixture
def depth_dataset(tmp_path: Path) -> Path:
    """A dataset config for a two-frame depth dataset and a scale config."""
    writer = DatasetWriter(
        tmp_path / "depth",
        name="synthetic_depth",
        type="depth",
        separator=None,
        meta={"radial_depth": False, "scale_to_meters": 1.0, "range": [0, 1000]},
        euler_loading={"loader": "generic_dense_depth", "function": "depth"},
    )
    for full_id in FRAMES:
        path = writer.get_path(full_id, full_id.rsplit("/", 1)[-1] + ".npy")
        write_depth(str(path), np.full((4, 6), 25.0, dtype=np.float32))
    writer.save_index()
    (tmp_path / "scale.json").write_text(json.dumps({"factor": 3.0}), encoding="utf-8")
    config = tmp_path / "dataset.json"
    config.write_text(
        json.dumps(
            {
                "transform_config_path": "scale.json",
                "output_path": str(tmp_path / "scaled"),
                "modalities": {"depth": str(tmp_path / "depth")},
            }
        ),
        encoding="utf-8",
    )
    return config


def _scaled_depths(root: Path) -> list[np.ndarray]:
    output = MultiModalDataset({"depth": Modality(str(root))})
    return [normalize_depth(output[i]["depth"]) for i in range(len(output))]


def test_builtin_transforms_come_from_the_catalog_with_provenance(monkeypatch):
    install(monkeypatch)
    entries = transform_catalog().entries
    assert list(entries) == sorted(BUILTIN_TRANSFORMS)
    fog = entries["fog"]
    assert fog.transform_class is FogTransform
    assert (fog.builtin, fog.extension.distribution) == (True, "euler-preprocess")
    assert transform_problems() == []
    assert run_options(FogTransform) == {
        "output_backends": True,
        "depth_radial": True,
        "strict": False,
        "samples_per_variant": True,
    }


def test_builtin_subcommands_run_the_registered_transform():
    args = build_parser().parse_args(["sky-depth", "-c", "dataset.json", "--strict"])
    assert (args.transform, args.strict, args.transform_config) == ("sky-depth", True, None)
    args = build_parser().parse_args(["run", "radial", "-c", "d.json", "-t", "r.json"])
    assert (args.transform, args.transform_config) == ("radial", "r.json")


def test_installed_transform_runs_through_the_generic_command(
    monkeypatch, depth_dataset, capsys
):
    install(monkeypatch, plugin("scale-depth", SCALE))
    entry = get_transform("scale-depth")
    assert entry.transform_class is ScaleDepthTransform
    assert (entry.builtin, entry.extension.distribution) == (False, "my-lab")

    assert main(["run", "scale-depth", "-c", str(depth_dataset)]) == 0
    scaled = _scaled_depths(depth_dataset.parent / "scaled")
    assert len(scaled) == len(FRAMES)
    for depth in scaled:
        np.testing.assert_allclose(depth, 75.0)
    head = MultiModalDataset(
        {"depth": Modality(str(depth_dataset.parent / "scaled"))}
    ).get_modality_index("depth")["head"]
    assert head["modality"]["meta"]["scaled"] is True

    assert main(["transforms"]) == 0
    assert "scale-depth" in capsys.readouterr().out


def test_transform_config_option_replaces_the_dataset_configs_path(
    monkeypatch, depth_dataset, tmp_path
):
    install(monkeypatch, plugin("scale-depth", SCALE))
    config = json.loads(depth_dataset.read_text())
    del config["transform_config_path"]
    depth_dataset.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="no transform_config_path"):
        run_transform("scale-depth", depth_dataset)

    other = tmp_path / "half.json"
    other.write_text(json.dumps({"factor": 0.5}), encoding="utf-8")
    run_transform("scale-depth", depth_dataset, transform_config=other)
    for depth in _scaled_depths(tmp_path / "scaled"):
        np.testing.assert_allclose(depth, 12.5)


def test_the_old_fog_config_path_key_is_not_read(monkeypatch, depth_dataset):
    install(monkeypatch, plugin("scale-depth", SCALE))
    config = json.loads(depth_dataset.read_text())
    config["fog_config_path"] = config.pop("transform_config_path")
    depth_dataset.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="no transform_config_path"):
        run_transform("scale-depth", depth_dataset)


def test_options_a_transform_does_not_accept_are_refused(monkeypatch, depth_dataset):
    install(monkeypatch, plugin("scale-depth", SCALE))
    with pytest.raises(ValueError, match="has no strict mode"):
        run_transform("scale-depth", depth_dataset, strict=True)
    config = json.loads(depth_dataset.read_text())
    config["file_ids"] = ["0000"]
    depth_dataset.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="file_ids selection is not supported"):
        run_transform("scale-depth", depth_dataset)


def test_unknown_transform_names_the_known_ones(monkeypatch, capsys):
    install(monkeypatch)
    assert main(["run", "nope", "-c", "dataset.json"]) == 2
    err = capsys.readouterr().err
    assert "unknown transform 'nope'" in err and "fog, lidar-motion, radial, sky-depth" in err


@pytest.mark.parametrize("name", ["fog", "crop"])
def test_installed_transform_cannot_take_a_builtin_name(monkeypatch, name, capsys):
    install(monkeypatch, plugin(name, "tests.plugin_transforms:NotFog"))
    assert _problems() == {name: ["shadows_builtin"]}
    if name == "fog":
        assert get_transform("fog").transform_class is FogTransform
    else:
        assert "crop" not in _names()
    assert main(["transforms"]) == 0
    out = capsys.readouterr().out
    assert "1 installed transform(s) not loaded" in out and "[shadows_builtin]" in out


def test_two_packages_offering_one_transform_is_a_reported_conflict(monkeypatch):
    install(
        monkeypatch,
        plugin("scale-depth", SCALE, "my-lab"),
        plugin("scale-depth", SCALE, "other-lab"),
    )
    assert "scale-depth" not in _names()
    assert _problems() == {"scale-depth": ["conflict"]}
    with pytest.raises(KeyError, match="all provide 'euler.transforms' extension"):
        get_transform("scale-depth")


@pytest.mark.parametrize(
    "name, target, message",
    [
        ("lab", "tests.plugin_transforms:NotATransform", "not a Transform subclass"),
        ("lab", "tests.plugin_transforms:Unfinished", "does not implement run"),
        ("lab", "tests.plugin_transforms:ForeignSource", "must be one of its REQUIRED_MODALITIES"),
        ("lab", "tests.plugin_transforms:AuxiliaryWithoutBackends", "must accept output_backends"),
        ("Lab Transform", SCALE, "must be lower-case letters"),
    ],
)
def test_invalid_transforms_are_reported(monkeypatch, name, target, message):
    install(monkeypatch, plugin(name, target))
    [problem] = transform_problems()
    assert problem.code == "invalid"
    assert message in problem.message
    assert _names() == sorted(BUILTIN_TRANSFORMS)


def test_a_broken_package_does_not_hide_the_builtins(monkeypatch):
    install(monkeypatch, plugin("lab", "tests.no_such_module:Lab"))
    assert _problems() == {"lab": ["load_failed"]}
    assert _names() == sorted(BUILTIN_TRANSFORMS)


def test_strict_mode_raises_on_problems(monkeypatch):
    monkeypatch.setenv("EULER_EXTENSIONS_STRICT", "1")
    install(monkeypatch, plugin("fog", "tests.plugin_transforms:NotFog"))
    with pytest.raises(ExtensionError, match="built-in transform"):
        transform_catalog()


def test_capability_report_describes_every_transform(monkeypatch, capsys):
    install(
        monkeypatch,
        plugin("scale-depth", SCALE),
        plugin("fog", "tests.plugin_transforms:NotFog", "other-lab"),
    )
    report = transform_capability_report()
    assert report["contract"] == {"kind": "transform_capability_report", "version": "1.0"}
    assert report["package"]["name"] == "euler-preprocess"
    assert report["extension_group"] == "euler.transforms"
    assert json.loads(json.dumps(report)) == report

    transforms = {entry["name"]: entry for entry in report["transforms"]}
    assert set(transforms) == set(BUILTIN_TRANSFORMS) | {"scale-depth"}
    scale = transforms["scale-depth"]
    assert scale["builtin"] is False
    assert scale["provider"]["distribution"] == "my-lab"
    assert scale["summary"] == "Multiply depth by a constant factor."
    assert scale["modalities"]["required"] == ["depth"]
    assert scale["outputs"] == [
        {"slot": "depth", "primary": True, "mirrors": "depth", "meta_overrides": {"scaled": True}}
    ]
    assert scale["config"]["factor"]["default"] == 2.0
    assert scale["command"] == "euler-preprocess run scale-depth -c <dataset-config>"

    fog = transforms["fog"]
    assert fog["builtin"] is True
    assert fog["modalities"]["optional"] == ["semantic_segmentation"]
    assert fog["modalities"]["loader_functions"] == {"semantic_segmentation": "sky_mask"}
    aux = {slot["slot"]: slot for slot in fog["outputs"] if not slot["primary"]}
    assert aux["scattering_coefficient"]["modality_key"] == "map_2d"
    assert aux["scattering_coefficient"]["opt_in"] is True
    assert fog["config"]["airlight"]["required"] is True
    assert transforms["sky-depth"]["run_options"]["strict"] is True
    assert transforms["radial"]["modalities"]["required_hierarchical"] == ["intrinsics"]

    assert {command["name"] for command in report["commands"]} >= {"crop", "spatial", "run"}
    assert [(p["name"], p["code"]) for p in report["problems"]] == [("fog", "shadows_builtin")]

    assert main(["transforms", "--json"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["contract"]["kind"] == "transform_capability_report"


def test_capability_report_is_registered_for_the_suite():
    eps = importlib.metadata.entry_points()
    group = (
        eps.select(group="euler.capabilities")
        if hasattr(eps, "select")
        else eps.get("euler.capabilities", ())
    )
    targets = {ep.name: ep.value for ep in group}
    assert targets["euler-preprocess"] == (
        "euler_preprocess.capabilities:transform_capability_report"
    )
    from euler_dataset_contract.suite import collect_capabilities

    capabilities, findings = collect_capabilities(only=["euler-preprocess"])
    assert capabilities["euler-preprocess"]["status"] == "ok"
    assert not [f for f in findings if f["subject"] == "euler-preprocess"]
