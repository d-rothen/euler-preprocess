from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from euler_preprocess.common.io import load_json
from euler_preprocess.common.logging import get_logger, progress_bar
from euler_preprocess.common.normalize import _to_numpy
from euler_preprocess.common.transform import Transform
from euler_preprocess.lidar_motion.compensation import compensate_ego_motion

_TIMING_KEYS = ("camera_exposure_start_us", "camera_exposure_end_us")
_GNSS_KEYS = ("ground_speed_m_s", "roll_rate_deg_s", "pitch_rate_deg_s", "heading_rate_deg_s")
_CONFIG_KEYS = {"timestamp_column"}


def _record(value: Any, name: str, keys: tuple[str, ...]) -> Mapping[str, Any]:
    """Return the sample's record, unwrapping a ``{file_id: record}`` level."""
    if isinstance(value, Mapping) and not all(key in value for key in keys) and len(value) == 1:
        value = next(iter(value.values()))
    if not isinstance(value, Mapping) or not all(key in value for key in keys):
        raise ValueError(
            f"Modality {name!r} must provide {', '.join(keys)}; got "
            f"{sorted(value) if isinstance(value, Mapping) else type(value).__name__}."
        )
    return value


def _matrix(value: Any, name: str) -> np.ndarray:
    if isinstance(value, Mapping):
        if len(value) != 1:
            raise ValueError(
                f"Sample must resolve exactly one {name!r} file, got {sorted(value)}."
            )
        value = next(iter(value.values()))
    matrix = _to_numpy(value).astype(np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name!r} must be a (4, 4) transform, got {matrix.shape}.")
    return matrix


class LidarMotionTransform(Transform):
    """Compensate scanning-lidar points for the vehicle's motion during a sweep.

    Every point moves to where it was at the frame camera's mid-exposure, so
    the point cloud lines up with the image (see
    :func:`~euler_preprocess.lidar_motion.compensation.compensate_ego_motion`).

    Modalities, all joined per sample by euler-loading:

    - ``sparse_depth``: the lidar points, ``(N, C)`` with each point's
      measurement time in seconds in column ``timestamp_column``;
    - ``gnss``: the frame's GNSS/IMU record with ``ground_speed_m_s``,
      ``roll_rate_deg_s``, ``pitch_rate_deg_s`` and ``heading_rate_deg_s``;
    - ``sensor_timing``: ``camera_exposure_start_us``/``camera_exposure_end_us``;
    - ``lidar_to_gnss``: the ``(4, 4)`` lidar → GNSS/IMU transform.

    For MUSES these come from ``muses.sparse_depth``, ``muses.gnss``,
    ``muses.sensor_timing`` and ``muses.read_lidar_to_gnss``. The GNSS frame is
    x forward, y right, z down, and the heading rate turns about its ``+z``.

    Config JSON keys:
        ``timestamp_column`` (int): column of the per-point time; default ``5``
        (MUSES: ``x, y, z, intensity, ring, timestamp``).

    Output: a ``sparse_depth`` dataset mirroring the input's paths and file
    format. The head records ``meta.motion_compensated``, and each file's
    ``attributes.motion_compensation`` records the target time and motion used.
    """

    REQUIRED_MODALITIES: ClassVar[set[str]] = {"sparse_depth", "gnss"}
    REQUIRED_HIERARCHICAL_MODALITIES: ClassVar[set[str]] = {"sensor_timing", "lidar_to_gnss"}
    SOURCE_MODALITY: ClassVar[str] = "sparse_depth"
    OUTPUT_SLOT: ClassVar[str] = "sparse_depth"
    OUTPUT_INDEX_META_OVERRIDES: ClassVar[dict[str, object]] = {
        "motion_compensated": {
            "target": "camera_mid_exposure",
            "model": "constant_speed_and_rates",
            "reference": "MUSES SDK processing/utils.py motion_compensate_pcd",
        }
    }

    def __init__(
        self,
        config_path: str,
        out_path: str,
        output_backend: Any | None = None,
    ) -> None:
        if output_backend is None or not getattr(output_backend, "is_source_backed", False):
            raise ValueError(
                "LidarMotionTransform writes through a source-backed output backend, "
                "so the lidar keeps its file format and float64 timestamps; run it "
                "with `euler-preprocess lidar-motion -c <config>`."
            )
        self.config_path = Path(config_path)
        self.output_backend = output_backend
        self.out_path = self.output_backend.root
        self.config = load_json(self.config_path)
        unknown = sorted(set(self.config) - _CONFIG_KEYS)
        if unknown:
            raise ValueError(f"Unknown lidar-motion config keys: {', '.join(unknown)}")
        column = self.config.get("timestamp_column", 5)
        if isinstance(column, bool) or not isinstance(column, int) or column < 3:
            raise ValueError("timestamp_column must be an integer >= 3")
        self.timestamp_column = column
        self.logger = get_logger()
        self.logger.info(
            "LidarMotionTransform: timestamp column %d, output=%s",
            self.timestamp_column,
            self.out_path,
        )

    def compensate(self, sample: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        """Return the sample's compensated points and the parameters used."""
        points = _to_numpy(sample["sparse_depth"]).astype(np.float64)
        if points.ndim != 2 or points.shape[1] <= self.timestamp_column:
            raise ValueError(
                f"Sample {sample.get('full_id', '?')!r}: sparse_depth must be (N, C) with "
                f"C > {self.timestamp_column}, got {points.shape}."
            )
        timing = _record(sample["sensor_timing"], "sensor_timing", _TIMING_KEYS)
        gnss = _record(sample["gnss"], "gnss", _GNSS_KEYS)
        lidar_to_gnss = _matrix(sample["lidar_to_gnss"], "lidar_to_gnss")
        target_us = (timing["camera_exposure_start_us"] + timing["camera_exposure_end_us"]) / 2
        compensated = points.copy()
        compensated[:, :3] = compensate_ego_motion(
            points[:, :3],
            points[:, self.timestamp_column],
            target_us / 1e6,
            lidar_to_gnss,
            ground_speed_m_s=gnss["ground_speed_m_s"],
            roll_rate_deg_s=gnss["roll_rate_deg_s"],
            pitch_rate_deg_s=gnss["pitch_rate_deg_s"],
            yaw_rate_deg_s=gnss["heading_rate_deg_s"],
        )
        parameters = {"target_time_us": target_us, **{key: float(gnss[key]) for key in _GNSS_KEYS}}
        return compensated, parameters

    def run(self, samples: Iterable[dict]) -> list[Path]:
        try:
            total = len(samples)  # type: ignore[arg-type]
        except TypeError:
            samples = list(samples)
            total = len(samples)
        saved_paths: list[Path] = []
        with progress_bar(total, "lidar-motion", self.logger) as bar:
            for sample in samples:
                compensated, parameters = self.compensate(sample)
                saved_paths.append(
                    self.output_backend.write(
                        sample,
                        compensated,
                        attributes={"motion_compensation": parameters},
                    )
                )
                if bar is not None:
                    bar.update(1)
        self.output_backend.finalize()
        return saved_paths
