"""Hierarchical intrinsics lookup."""

from __future__ import annotations

import numpy as np
import pytest

from euler_preprocess.common.intrinsics import extract_intrinsics, single_calibration


def test_the_file_named_intrinsics_wins():
    assert single_calibration({"intrinsics": 1, "other": 2}) == 1


def test_a_single_file_is_used_whatever_its_id():
    """SeeingThroughFog names its camera calibration ``calib_cam_stereo_left``."""
    K = np.diag([2355.7, 2355.7, 1.0]).astype(np.float32)
    np.testing.assert_array_equal(extract_intrinsics({"intrinsics": {"calib_cam_stereo_left": K}}), K)


def test_several_files_are_ambiguous():
    with pytest.raises(ValueError, match="matched 2 files .*a, b"):
        single_calibration({"a": 1, "b": 2})


def test_missing_and_per_sample_values():
    assert extract_intrinsics({}) is None
    assert single_calibration({}) is None
    K = np.eye(3)
    np.testing.assert_array_equal(extract_intrinsics({"intrinsics": K}), K.astype(np.float32))
