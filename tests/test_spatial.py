from copy import deepcopy

import numpy as np
import pytest
from ds_crawler import read_record
from euler_dataset_contract import resolve_receipt
from euler_dataset_contract.testing.checks._phase2_support import make_source
from euler_loading import CalibrationView, Modality, MultiModalDataset

from euler_preprocess.common.dataset import build_dataset
from euler_preprocess.spatial import export_spatial


def config_for(tmp_path, *, zip_output=False):
    ds, plan, data, original = make_source(tmp_path / "source", tiny=True)
    config = {
        "modalities": {
            field: str(tmp_path / "source" / field)
            for field in plan["recipe"]["fields"]
            if field != "intrinsics"
        },
        "hierarchical_modalities": {"intrinsics": str(tmp_path / "source/intrinsics")},
        "preprocessing": data["config"],
        "bindings": {
            "fields": plan["recipe"]["fields"],
            "image_planes": plan["recipe"]["image_planes"],
            "revisions": {a: s["revision"] for a, s in plan["sources"].items()},
        },
        "outputs": {
            field: {
                "path": str(tmp_path / (field + (".zip" if zip_output else "_out"))),
                "dataset_id": "derived_" + field,
                "revision": "v1",
                "metadata_scope": "saved",
            }
            for field in plan["recipe"]["fields"]
        },
        "receipt_sidecars": True,
        "samples": [1],
        "mapping": {"default": {"/scene_01/camera_0/frame_1": "/variant/crop_a"}},
    }
    return config, ds, plan, data, original


@pytest.mark.parametrize("zip_output", [False, True])
def test_producer_captures_and_reloads_five_outputs(tmp_path, zip_output):
    config, ds, plan, _data, original = config_for(tmp_path, zip_output=zip_output)
    paths = export_spatial(config)
    ds.enable_transform_capture({"default": plan})
    expected = ds[1]
    for field, path in paths.items():
        dataset = MultiModalDataset({field: Modality(path, metadata_scope="saved")})
        assert len(dataset) == 1
        sample = dataset[0]
        assert sample["full_id"] == "/variant/crop_a"
        np.testing.assert_array_equal(sample[field], expected[field])
        attrs = sample["attributes"][field]
        assert attrs["note"] == "preserved"
        receipt = resolve_receipt(
            attrs["euler_transforms"],
            lambda p, root=path: read_record(root, p, metadata_scope="saved"),
        )
        assert (
            receipt["execution"]["source_full_ids"]["source_rgb"]
            == "/scene_01/camera_0/frame_1"
        )
        head = dataset.get_modality_index(field)["head"]
        frozen = head["addons"]["euler_transforms"]["plan"]
        assert frozen["field"] == field
        assert field in frozen["dependencies"]["default"]
        view = CalibrationView(
            frozen["variants"]["default"], receipt["execution"], "intrinsics"
        )
        np.testing.assert_array_equal(
            view.resolve(next(iter(original["intrinsics"].values()))),
            expected["intrinsics"],
        )


def test_producer_infers_hierarchical_intrinsics_listed_as_modality(tmp_path):
    config, ds, plan, _data, _original = config_for(tmp_path)
    config["modalities"].update(config.pop("hierarchical_modalities"))
    paths = export_spatial(config)
    ds.enable_transform_capture({"default": plan})
    expected = ds[1]
    for field, path in paths.items():
        sample = MultiModalDataset({field: Modality(path, metadata_scope="saved")})[0]
        np.testing.assert_array_equal(sample[field], expected[field])


def test_common_builder_consumes_preprocessing(tmp_path):
    config, ds, _, _, _ = config_for(tmp_path)
    configured = build_dataset(config, set(config["modalities"]))
    assert configured[0]["rgb"].shape == (2, 2, 3)
    assert ds[0]["rgb"].shape == (4, 8, 3)


def test_variants_and_resume(tmp_path):
    config, _, _, _, _ = config_for(tmp_path)
    first = {
        "preprocessing": config.pop("preprocessing"),
        "bindings": config.pop("bindings"),
    }
    second = deepcopy(first)
    second["preprocessing"] = {
        "crop": {"size": [3, 3], "offset": [1, 1]},
        "fields": first["preprocessing"]["fields"],
        "reference_field": "rgb",
        "infer_fields": False,
    }
    config["spatial_variants"] = {"small": first, "large": second}
    config.pop("mapping")
    config["outputs"] = {"rgb": config["outputs"]["rgb"]}
    paths = export_spatial(config)
    dataset = MultiModalDataset({"rgb": Modality(paths["rgb"], metadata_scope="saved")})
    assert len(dataset) == 2
    assert (
        "dimensions"
        not in dataset.get_modality_index("rgb")["head"]["modality"]["meta"]
    )
    assert {dataset[i]["full_id"].rsplit("/", 1)[-1] for i in range(2)} == {
        "small",
        "large",
    }
    config["outputs"]["rgb"]["resume"] = True
    export_spatial(config)
    assert (
        len(MultiModalDataset({"rgb": Modality(paths["rgb"], metadata_scope="saved")}))
        == 2
    )
