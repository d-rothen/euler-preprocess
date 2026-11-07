"""Tests for the lidar-motion transform (scanning-lidar ego-motion compensation)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
from ds_crawler import DatasetWriter
from euler_loading import Modality, MultiModalDataset
from euler_loading.loaders.cpu import muses

from euler_preprocess.cli import _run_transform
from euler_preprocess.lidar_motion.compensation import compensate_ego_motion
from euler_preprocess.lidar_motion.transform import LidarMotionTransform

needs_muses_motion_inputs = pytest.mark.skipif(
    not hasattr(muses, "sensor_timing"),
    reason="needs euler-loading with the MUSES gnss/sensor_timing/read_lidar_to_gnss loaders",
)

# MUSES-like lidar -> GNSS mount: GNSS x forward, y right, z down.
LIDAR_TO_GNSS = np.array(
    [[1.0, 0.0, 0.0, 0.048], [0.0, -1.0, 0.0, 0.286], [0.0, 0.0, -1.0, -0.039], [0.0, 0.0, 0.0, 1.0]]
)


def _rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return rz @ ry @ rx


MOTION = dict(ground_speed_m_s=8.0, roll_rate_deg_s=1.0, pitch_rate_deg_s=-0.5, yaw_rate_deg_s=-25.0)


class TestCompensation:
    def test_no_motion_changes_nothing(self):
        xyz = np.random.default_rng(0).normal(size=(10, 3)) * 10
        out = compensate_ego_motion(
            xyz, np.linspace(0.0, 0.08, 10), 0.04, LIDAR_TO_GNSS,
            ground_speed_m_s=0.0, roll_rate_deg_s=0.0, pitch_rate_deg_s=0.0, yaw_rate_deg_s=0.0,
        )
        np.testing.assert_allclose(out, xyz, atol=1e-12)

    def test_points_measured_at_the_target_time_stay(self):
        xyz = np.array([[10.0, 2.0, 1.0], [20.0, -3.0, 0.5]])
        out = compensate_ego_motion(xyz, np.array([0.04, 0.04]), 0.04, LIDAR_TO_GNSS, **MOTION)
        np.testing.assert_allclose(out, xyz, atol=1e-12)

    def test_forward_motion_pulls_earlier_points_closer(self):
        """Measured 50 ms before the target at 10 m/s: the car was 0.5 m further back."""
        out = compensate_ego_motion(
            np.array([[20.0, 0.0, 0.0]]), np.array([0.0]), 0.05, np.eye(4),
            ground_speed_m_s=10.0, roll_rate_deg_s=0.0, pitch_rate_deg_s=0.0, yaw_rate_deg_s=0.0,
        )
        np.testing.assert_allclose(out, [[19.5, 0.0, 0.0]], atol=1e-12)

    def test_rotation_matches_explicit_matrices_in_the_ego_frame(self):
        rng = np.random.default_rng(1)
        xyz = rng.normal(size=(50, 3)) * 15
        times = rng.uniform(0.0, 0.085, 50)
        out = compensate_ego_motion(xyz, times, 0.05, LIDAR_TO_GNSS, **MOTION)

        ego = xyz @ LIDAR_TO_GNSS[:3, :3].T + LIDAR_TO_GNSS[:3, 3]
        inverse = np.linalg.inv(LIDAR_TO_GNSS)
        for point, t, actual in zip(ego, times, out):
            dt = t - 0.05
            shifted = point + [MOTION["ground_speed_m_s"] * dt, 0.0, 0.0]
            rotated = _rotation(*np.deg2rad([MOTION["roll_rate_deg_s"], MOTION["pitch_rate_deg_s"], MOTION["yaw_rate_deg_s"]]) * dt) @ shifted
            np.testing.assert_allclose(actual, inverse[:3, :3] @ rotated + inverse[:3, 3], atol=1e-9)

    def test_recovers_a_static_scene_from_a_moving_turning_car(self):
        """Simulate the sweep the model describes, then undo it."""
        rng = np.random.default_rng(2)
        scene_ego = rng.uniform([5, -10, -2], [40, 10, 3], size=(200, 3))  # at the target time
        times = rng.uniform(0.0, 0.085, 200)
        target = 0.06
        rates = np.deg2rad([MOTION["roll_rate_deg_s"], MOTION["pitch_rate_deg_s"], MOTION["yaw_rate_deg_s"]])
        measured_ego = np.array([
            _rotation(*(rates * (t - target))).T @ p - [MOTION["ground_speed_m_s"] * (t - target), 0.0, 0.0]
            for p, t in zip(scene_ego, times)
        ])
        inverse = np.linalg.inv(LIDAR_TO_GNSS)
        measured = measured_ego @ inverse[:3, :3].T + inverse[:3, 3]

        out = compensate_ego_motion(measured, times, target, LIDAR_TO_GNSS, **MOTION)
        expected = scene_ego @ inverse[:3, :3].T + inverse[:3, 3]
        np.testing.assert_allclose(out, expected, atol=1e-9)

    def test_rejects_mismatched_inputs(self):
        with pytest.raises(ValueError, match=r"\(N, 3\)"):
            compensate_ego_motion(np.zeros((4, 4)), np.zeros(4), 0.0, np.eye(4), **MOTION)
        with pytest.raises(ValueError, match="point_times_s"):
            compensate_ego_motion(np.zeros((4, 3)), np.zeros(3), 0.0, np.eye(4), **MOTION)


# ---------------------------------------------------------------------------
# End to end through the CLI, on MUSES-like synthetic datasets
# ---------------------------------------------------------------------------

FRAMES = {"REC0001_frame_000101": "/test/fog/day/000101", "REC0001_frame_000102": "/test/fog/day/000102"}
SWEEP_START = 1676367085.705051


def _gnss_line(speed_mm_s: int, heading_rate: int) -> str:
    values = [0] * 28
    values[1] = int(SWEEP_START * 1e6)
    values[17], values[18], values[19] = 35000, -2000, heading_rate  # roll, pitch, heading rates
    values[23] = speed_mm_s
    return " ".join(str(v) for v in values) + "\n"


def _writer(root: Path, key: str, function: str) -> DatasetWriter:
    return DatasetWriter(
        root, name=f"muses_{key}", type=key, separator=None,
        euler_loading={"loader": "muses", "function": function},
    )


@pytest.fixture()
def muses_dataset(tmp_path):
    rng = np.random.default_rng(3)
    lidar = _writer(tmp_path / "lidar", "sparse_depth", "sparse_depth")
    gnss = _writer(tmp_path / "gnss", "gnss", "gnss")
    clouds, meta = {}, {}
    for i, (frame, full_id) in enumerate(FRAMES.items()):
        cloud = np.column_stack([
            rng.uniform([5, -10, -2], [40, 10, 3], size=(64, 3)),
            rng.uniform(0, 255, 64),
            rng.integers(1, 6, 64),
            SWEEP_START + np.sort(rng.uniform(0.0, 0.085, 64)),
        ])
        muses.write_sparse_depth(str(lidar.get_path(full_id, f"{frame}_lidar.bin")), cloud)
        gnss.get_path(full_id, f"{frame}_gnss.txt").write_text(
            _gnss_line(speed_mm_s=6000 + 2000 * i, heading_rate=-1672000 * (i + 1)), encoding="utf-8"
        )
        clouds[full_id] = cloud
        cam_start = int((SWEEP_START + 0.034 + 0.02 * i) * 1e6)
        meta[frame] = {
            "frame_camera_exposure_start_timestamp_us": str(cam_start),
            "frame_camera_exposure_end_timestamp_us": str(cam_start + 700),
            "lidar_start_timestamp_us": str(int(SWEEP_START * 1e6)),
            "lidar_end_timestamp_us": str(int((SWEEP_START + 0.0847) * 1e6)),
            "path_to_lidar": f"lidar/test/fog/day/{frame}_lidar.bin",
            "path_to_gnss": f"gnss/test/fog/day/{frame}_gnss.txt",
        }
    lidar.save_index()
    gnss.save_index()

    timing = _writer(tmp_path / "timing", "sensor_timing", "sensor_timing")
    timing.get_path("/meta", "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    timing.save_index()
    calib = _writer(tmp_path / "calib", "lidar_to_gnss", "read_lidar_to_gnss")
    calib.get_path("/calib", "calib.json").write_text(
        json.dumps({"extrinsics": {"lidar2gnss": LIDAR_TO_GNSS.tolist(), "lidar2rgb": np.eye(4).tolist()}}),
        encoding="utf-8",
    )
    calib.save_index()

    transform_config = tmp_path / "lidar_motion_config.json"
    transform_config.write_text(json.dumps({"timestamp_column": 5}), encoding="utf-8")
    dataset_config = tmp_path / "dataset_config.json"
    dataset_config.write_text(json.dumps({
        "transform_config_path": transform_config.name,
        "output_path": str(tmp_path / "compensated"),
        "modalities": {
            "sparse_depth": str(tmp_path / "lidar"),
            "gnss": str(tmp_path / "gnss"),
            "sensor_timing": str(tmp_path / "timing"),
            "lidar_to_gnss": str(tmp_path / "calib"),
        },
    }), encoding="utf-8")
    return dataset_config, tmp_path / "compensated", clouds, meta


@needs_muses_motion_inputs
def test_cli_writes_compensated_lidar(muses_dataset):
    dataset_config, out_root, clouds, meta = muses_dataset
    assert _run_transform(argparse.Namespace(config=str(dataset_config), strict=False), LidarMotionTransform) == 0

    out = MultiModalDataset(modalities={"sparse_depth": Modality(str(out_root))})
    assert len(out) == len(FRAMES)
    head_meta = out.get_modality_index("sparse_depth")["head"]["modality"]["meta"]
    assert head_meta["motion_compensated"]["target"] == "camera_mid_exposure"
    for i in range(len(out)):
        sample = out[i]
        frame = "REC0001_frame_" + sample["id"]
        assert sample["meta"]["sparse_depth"]["path"] == f"test/fog/day/{frame}_lidar.bin"
        source = clouds[sample["full_id"]]
        record = meta[frame]
        target = (int(record["frame_camera_exposure_start_timestamp_us"]) + int(record["frame_camera_exposure_end_timestamp_us"])) / 2
        gnss = muses.gnss(str(out_root.parent / "gnss" / "test/fog/day" / f"{frame}_gnss.txt"))
        expected = source.copy()
        expected[:, :3] = compensate_ego_motion(
            source[:, :3], source[:, 5], target / 1e6, LIDAR_TO_GNSS,
            ground_speed_m_s=gnss["ground_speed_m_s"], roll_rate_deg_s=gnss["roll_rate_deg_s"],
            pitch_rate_deg_s=gnss["pitch_rate_deg_s"], yaw_rate_deg_s=gnss["heading_rate_deg_s"],
        )
        actual = np.asarray(sample["sparse_depth"])
        np.testing.assert_allclose(actual, expected, atol=1e-9)
        np.testing.assert_array_equal(actual[:, 3:], source[:, 3:])  # intensity, ring, timestamp kept
        assert not np.allclose(actual[:, :3], source[:, :3])
        parameters = sample["attributes"]["sparse_depth"]["motion_compensation"]
        assert parameters["target_time_us"] == target
        assert parameters["ground_speed_m_s"] == pytest.approx(gnss["ground_speed_m_s"])


def test_transform_needs_a_source_backed_output(tmp_path):
    config = tmp_path / "c.json"
    config.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="source-backed"):
        LidarMotionTransform(str(config), str(tmp_path / "out"))


def test_unknown_config_keys_are_rejected(tmp_path):
    class Backend:
        is_source_backed = True
        root = tmp_path

    config = tmp_path / "c.json"
    config.write_text(json.dumps({"timestamp_col": 5}), encoding="utf-8")
    with pytest.raises(ValueError, match="timestamp_col"):
        LidarMotionTransform(str(config), str(tmp_path), output_backend=Backend())
