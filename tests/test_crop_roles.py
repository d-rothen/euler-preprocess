"""Crop roles come from loader annotations, never from modality names."""

from __future__ import annotations

import numpy as np
import pytest
from euler_loading.loaders.cpu import princeton_dense

from euler_preprocess.crop.roles import COPY, INTRINSICS, RASTER, infer_role


def _role(loader, *, hierarchical=False, name="x", override=None):
    return infer_role(name, loader._modality_meta, hierarchical=hierarchical, override=override)


def test_seeing_through_fog_loaders():
    assert _role(princeton_dense.rgb).role == RASTER
    assert _role(princeton_dense.rgb).layout == "HWC"
    assert _role(princeton_dense.sparse_depth, name="sparse_depth").role == COPY
    assert _role(princeton_dense.read_intrinsics, hierarchical=True).role == INTRINSICS
    assert _role(princeton_dense.read_extrinsics, hierarchical=True).role == COPY


@pytest.mark.parametrize(
    ("shape", "modality_type", "role"),
    [
        ("HW", "depth", RASTER),
        ("1HW", "depth", RASTER),
        ("CHW", "rgb", RASTER),
        ("3HW", "rgb", RASTER),
        ("NC", "sparse_depth", COPY),
        ("4x4", "camera_extrinsics", COPY),
        ("Nx1", "extrinsics", COPY),
        ("3x3", "intrinsics", INTRINSICS),
    ],
)
def test_role_follows_the_annotated_shape(shape, modality_type, role):
    # The name is a decoy: a depth-like name never makes a point cloud a raster.
    assert infer_role("sparse_depth", {"shape": shape, "type": modality_type}, hierarchical=False).role == role


def test_unknown_shapes_need_an_override():
    annotation = {"shape": "dict", "type": "calibration"}
    with pytest.raises(ValueError, match='"roles": {"calibration": "raster" \\| "copy"}'):
        infer_role("calibration", annotation, hierarchical=True)
    assert infer_role("calibration", annotation, hierarchical=True, override="copy").role == COPY
    with pytest.raises(ValueError, match="cannot be cropped as a raster"):
        infer_role("calibration", annotation, hierarchical=True, override="raster")


def test_a_3x3_matrix_that_is_not_intrinsics_is_not_cropped_as_one():
    with pytest.raises(ValueError, match="Don't know how to crop"):
        infer_role("rotation", {"shape": "3x3", "type": "rotation"}, hierarchical=False)


def test_unannotated_loaders_need_an_override():
    with pytest.raises(ValueError, match="no @modality_meta annotation"):
        infer_role("custom", None, hierarchical=False)


def test_slicing_follows_the_layout():
    hwc = _role(princeton_dense.rgb)
    image = np.arange(4 * 5 * 3).reshape(4, 5, 3)
    cropped = hwc.slice(image, 1, 2, 2, 3)
    np.testing.assert_array_equal(cropped, image[1:3, 2:5])
    assert cropped.flags["C_CONTIGUOUS"]
    assert hwc.raster_size(image) == (4, 5)

    chw = infer_role("rgb", {"shape": "CHW", "type": "rgb"}, hierarchical=False)
    tensor = np.arange(3 * 4 * 5).reshape(3, 4, 5)
    np.testing.assert_array_equal(chw.slice(tensor, 1, 2, 2, 3), tensor[:, 1:3, 2:5])
    assert chw.raster_size(tensor) == (4, 5)

    with pytest.raises(ValueError, match="annotated HWC but loaded a value of shape"):
        hwc.raster_size(np.zeros((4, 5)))


def test_torch_rasters_are_sliced_in_chw():
    torch = pytest.importorskip("torch")
    role = infer_role("rgb", {"shape": "CHW", "type": "rgb"}, hierarchical=False)
    tensor = torch.arange(3 * 4 * 5).reshape(3, 4, 5)
    cropped = role.slice(tensor, 1, 1, 2, 2)
    assert cropped.is_contiguous()
    assert torch.equal(cropped, tensor[:, 1:3, 1:3])
