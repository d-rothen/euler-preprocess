"""Crop config parsing and crop-box resolution."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from euler_preprocess.crop.spec import CropBox, parse_crop_config, parse_crop_spec


def _box(spec: dict, height: int, width: int) -> tuple[int, int, int, int]:
    box = parse_crop_spec(spec).resolve(height, width)
    return box.top, box.left, box.height, box.width


class TestRemove:
    def test_fractions_round_half_up_per_edge(self):
        # 0.25 * 10 = 2.5 rounds up to 3 (Python's round() would give 2).
        spec = {"remove": {"top": 0.25, "bottom": 0.05, "left": 0.15, "right": 0.19}}
        assert _box(spec, 10, 20) == (3, 3, 6, 13)

    def test_pixels_reproduce_the_official_rgb2gatedv2_crop(self):
        spec = {"remove": {"top": 202, "bottom": 54, "left": 280, "right": 360}, "unit": "pixel"}
        assert _box(spec, 1024, 1920) == (202, 280, 768, 1280)

    def test_stf_valid_rectification_region(self):
        spec = {"remove": {"top": 34, "bottom": 46, "left": 16}, "unit": "pixel"}
        box = parse_crop_spec(spec).resolve(1024, 1920)
        assert box == CropBox(34, 16, 944, 1904, 1024, 1920)
        assert (box.bottom, box.right) == (46, 0)

    def test_multiple_of_trims_evenly_inside_the_requested_region(self):
        spec = {"remove": {"top": 0.1}, "multiple_of": 32}
        # Kept 90 rows -> 64; the 26 extra rows split 13/13. 100 columns -> 96, 2/2.
        assert _box(spec, 100, 100) == (10 + 13, 2, 64, 96)

    def test_removing_every_pixel_is_an_error(self):
        spec = parse_crop_spec({"remove": {"top": 60, "bottom": 40}, "unit": "pixel"})
        with pytest.raises(ValueError, match="leaves no pixels"):
            spec.resolve(100, 100)


class TestKeep:
    @pytest.mark.parametrize(
        ("anchor", "expected"),
        [
            ("center", (12, 16, 75, 67)),
            ("bottom", (25, 16, 75, 67)),
            ("top_left", (0, 0, 75, 67)),
            ("right", (12, 33, 75, 67)),
            ("bottom_right", (25, 33, 75, 67)),
        ],
    )
    def test_anchors(self, anchor, expected):
        spec = {"keep": {"height": 0.75, "width": 0.67}, "anchor": anchor}
        assert _box(spec, 100, 100) == expected

    def test_omitted_side_keeps_the_full_axis(self):
        assert _box({"keep": {"height": 64}, "unit": "pixel"}, 100, 80) == (18, 0, 64, 80)

    def test_center_matches_euler_loading_crop(self):
        from euler_loading.preprocessing import Crop

        top, left, height, width = _box({"keep": {"height": 37, "width": 51}, "unit": "pixel"}, 100, 80)
        assert (top, left, height, width) == Crop(size=(37, 51)).resolve_box((100, 80))

    def test_more_pixels_than_the_reference_is_an_error(self):
        with pytest.raises(ValueError, match="Cannot keep 120 px"):
            parse_crop_spec({"keep": {"width": 120}, "unit": "pixel"}).resolve(100, 100)


class TestValidation:
    @pytest.mark.parametrize(
        ("spec", "message"),
        [
            ({"remove": {"top": 10}}, r"fraction in \[0, 1\).*unit 'pixel'"),
            ({"keep": {"height": 75}}, r"fraction in \(0, 1\]"),
            ({"remove": {"top": 0.6, "bottom": 0.4}}, "must be below 1"),
            ({"remove": {"top": 0.1}, "keep": {"height": 0.5}}, "exactly one of"),
            ({"remove": {"top": 0.1}, "anchor": "top"}, "only applies to 'keep'"),
            ({"remove": {"top": True}}, "must be a number"),
            ({"remove": {"top": 1.5}, "unit": "pixel"}, "whole number"),
            ({"remove": {"middle": 0.1}}, "unknown keys: middle"),
            ({"keep": {"height": 0.5}, "anchor": "middle"}, "anchor must be one of"),
            ({"remove": {"top": 0.1}, "multiple_of": 0}, "positive integer"),
            ({"remove": {}}, "non-empty object"),
        ],
    )
    def test_rejected_specs(self, spec, message):
        with pytest.raises(ValueError, match=message):
            parse_crop_spec(spec)

    def test_config_defaults(self):
        config = parse_crop_config({"crop": {"remove": {"top": 0.1}}})
        assert (config.reference, config.point_clouds, config.verify) == ("rgb", "keep", "all")

    @pytest.mark.parametrize(
        ("raw", "message"),
        [
            ({}, "needs a 'crop' entry"),
            ({"crop": {"remove": {"top": 0.1}}, "preprocessing": {}}, "would crop twice"),
            ({"crop": {"remove": {"top": 0.1}}, "point_clouds": "drop"}, "point_clouds"),
            ({"crop": {"remove": {"top": 0.1}}, "verify": 0}, "verify"),
            ({"crop": {"remove": {"top": 0.1}}, "roles": {"depth": "resize"}}, "roles.depth"),
            ({"crop": {"remove": {"top": 0.1}}, "scale": 2}, "Unknown crop config keys: scale"),
        ],
    )
    def test_rejected_configs(self, raw, message):
        with pytest.raises(ValueError, match=message):
            parse_crop_config(raw)

    def test_spec_round_trips_through_to_dict(self):
        spec = parse_crop_spec({"keep": {"height": 0.5}, "anchor": "bottom", "multiple_of": 8})
        assert parse_crop_spec(spec.to_dict()) == spec


def test_shipped_configs():
    configs = Path(__file__).resolve().parents[1] / "configs"
    fractions = parse_crop_config(json.loads((configs / "crop_config.json").read_text()))
    assert fractions.crop.resolve(1024, 1920) == CropBox(205, 288, 768, 1267, 1024, 1920)
    stf = parse_crop_config(json.loads((configs / "crop_stf_valid_region_config.json").read_text()))
    assert stf.crop.resolve(1024, 1920) == CropBox(34, 16, 944, 1904, 1024, 1920)
    dataset_config = json.loads((configs / "crop_stf_dataset_config.json").read_text())
    assert dataset_config["transform_config_path"] == "crop_stf_valid_region_config.json"


class TestResize:
    def test_scale_halves_the_crop_like_sweden2psmnet(self):
        config = parse_crop_config({"crop": {"keep": {"height": 1.0}}, "resize": {"scale": 0.5}})
        box = config.geometry(1024, 1920)
        assert (box.height, box.width, box.output_height, box.output_width) == (1024, 1920, 512, 960)
        assert box.resized

    def test_size_and_per_axis_scale(self):
        crop = {"remove": {"top": 202, "bottom": 54, "left": 280, "right": 360}, "unit": "pixel"}
        box = parse_crop_config({"crop": crop, "resize": {"size": [384, 640]}}).geometry(1024, 1920)
        assert (box.output_height, box.output_width) == (384, 640)
        box = parse_crop_config({"crop": crop, "resize": {"scale": [0.5, 0.25]}}).geometry(1024, 1920)
        assert (box.output_height, box.output_width) == (384, 320)

    def test_pixel_transform_is_resize_after_crop(self):
        from euler_loading.geometry import crop_matrix, resize_matrix

        config = parse_crop_config({"crop": {"remove": {"top": 10, "left": 20}, "unit": "pixel"}, "resize": {"scale": 0.5}})
        box = config.geometry(100, 200)
        expected = np.asarray(resize_matrix((90, 180), (45, 90))) @ np.asarray(crop_matrix(10, 20))
        np.testing.assert_allclose(np.asarray(box.pixel_transform), expected, atol=1e-12)
        unresized = parse_crop_config({"crop": {"remove": {"top": 10, "left": 20}, "unit": "pixel"}}).geometry(100, 200)
        np.testing.assert_array_equal(np.asarray(unresized.pixel_transform), np.asarray(crop_matrix(10, 20)))

    @pytest.mark.parametrize(
        ("resize", "message"),
        [
            ({"size": [384, 640], "scale": 0.5}, "exactly one of"),
            ({}, "exactly one of"),
            ({"scale": 0}, "positive"),
            ({"size": 384}, r"\[height, width\]"),
            ({"size": [384.5, 640]}, "whole pixels"),
            ({"scale": 0.5, "interpolation": {"depth": "area"}}, "interpolation.depth"),
            ({"scale": 0.5, "mode": "fast"}, "unknown keys: mode"),
        ],
    )
    def test_rejected_resizes(self, resize, message):
        with pytest.raises(ValueError, match=message):
            parse_crop_config({"crop": {"remove": {"top": 0.1}}, "resize": resize})

    def test_projection_needs_pruning(self):
        with pytest.raises(ValueError, match="only applies to point_clouds 'prune_to_crop'"):
            parse_crop_config({"crop": {"remove": {"top": 0.1}}, "projection": {"intrinsics": "K"}})
        config = parse_crop_config({
            "crop": {"remove": {"top": 0.1}},
            "point_clouds": "prune_to_crop",
            "projection": {"intrinsics": "intrinsics", "extrinsics": "camera_extrinsics"},
        })
        assert config.projection == {"intrinsics": "intrinsics", "extrinsics": "camera_extrinsics"}


def test_spatial_preprocessing_reproduces_the_box():
    from euler_loading import SamplePreprocessor

    from euler_preprocess.crop.spec import spatial_preprocessing

    config = parse_crop_config({"crop": {"remove": {"top": 0.25, "left": 0.1}}, "resize": {"scale": 0.5}})
    box = config.geometry(40, 60)
    block = spatial_preprocessing(box)
    assert block["operations"] == [
        {"type": "crop", "size": [30, 54], "offset": [10, 6]},
        {"type": "resize", "size": [15, 27]},
    ]
    image = np.arange(40 * 60, dtype=np.float32).reshape(40, 60)
    preprocessor = SamplePreprocessor.from_config({**block, "fields": {"rgb": {"kind": "image", "layout": "HW"}}, "infer_fields": False})
    assert tuple(preprocessor({"rgb": image})["rgb"].shape) == (15, 27)
