"""Ego-motion compensation for scanning lidar.

A scanning lidar measures its points one after another while the vehicle
moves; the MUSES lidar sweeps its field of view top to bottom in about 85 ms.
Projected into a camera image without correction, tall objects match at the
height that was scanned during the exposure and drift away above and below it.
"""

from __future__ import annotations

import numpy as np


def compensate_ego_motion(
    xyz: np.ndarray,
    point_times_s: np.ndarray,
    target_time_s: float,
    sensor_to_ego: np.ndarray,
    *,
    ground_speed_m_s: float,
    roll_rate_deg_s: float,
    pitch_rate_deg_s: float,
    yaw_rate_deg_s: float,
) -> np.ndarray:
    """Move lidar points to where they were at ``target_time_s``.

    This follows the official MUSES SDK (``processing/utils.py``,
    ``motion_compensate_pcd``). The points go into the ego frame, the GNSS/IMU
    frame the motion is measured in. There the vehicle drives along ``+x`` at
    a constant ground speed and turns at constant body rates. A point measured
    ``dt = t - target_time_s`` seconds from the target is shifted by
    ``ground_speed * dt`` along ``x``. It is then rotated by
    ``Rz(yaw_rate * dt) @ Ry(pitch_rate * dt) @ Rx(roll_rate * dt)`` and
    taken back to the sensor frame.

    Args:
        xyz: ``(N, 3)`` points in the sensor frame, in metres.
        point_times_s: ``(N,)`` measurement time of each point, in seconds.
        target_time_s: Time the points are moved to, in seconds, e.g. the
            camera's mid-exposure.
        sensor_to_ego: ``(4, 4)`` transform from the sensor to the ego frame.
        ground_speed_m_s: Forward speed along the ego ``x`` axis.
        roll_rate_deg_s, pitch_rate_deg_s, yaw_rate_deg_s: Rotation rates
            about the ego ``x``, ``y`` and ``z`` axes.

    Returns:
        ``(N, 3)`` float64 points in the sensor frame.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must have shape (N, 3), got {xyz.shape}")
    dt = np.asarray(point_times_s, dtype=np.float64) - float(target_time_s)
    if dt.shape != (len(xyz),):
        raise ValueError(f"point_times_s must have shape ({len(xyz)},), got {dt.shape}")
    transform = np.asarray(sensor_to_ego, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"sensor_to_ego must have shape (4, 4), got {transform.shape}")

    ego = xyz @ transform[:3, :3].T + transform[:3, 3]
    ego[:, 0] += float(ground_speed_m_s) * dt

    roll = np.deg2rad(float(roll_rate_deg_s)) * dt
    pitch = np.deg2rad(float(pitch_rate_deg_s)) * dt
    yaw = np.deg2rad(float(yaw_rate_deg_s)) * dt
    sr, cr = np.sin(roll), np.cos(roll)
    sp, cp = np.sin(pitch), np.cos(pitch)
    sy, cy = np.sin(yaw), np.cos(yaw)
    x, y, z = ego[:, 0], ego[:, 1], ego[:, 2]
    rotated = np.column_stack(
        [
            cy * cp * x + (cy * sp * sr - sy * cr) * y + (cy * sp * cr + sy * sr) * z,
            sy * cp * x + (sy * sp * sr + cy * cr) * y + (sy * sp * cr - cy * sr) * z,
            -sp * x + cp * sr * y + cp * cr * z,
        ]
    )

    inverse = np.linalg.inv(transform)
    return rotated @ inverse[:3, :3].T + inverse[:3, 3]
