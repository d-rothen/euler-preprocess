from __future__ import annotations

import numpy as np

from euler_preprocess.common.normalize import _to_numpy

try:
    import torch
except ImportError:
    torch = None


def single_calibration(value, *, name: str = "intrinsics"):
    """Return the one calibration value a sample carries for modality *name*.

    Hierarchical modalities arrive as ``{file_id: value}``. A file whose ID is
    *name* wins; otherwise the dict must hold exactly one file (e.g. STF's
    ``calib_cam_stereo_left``). Several files are ambiguous and raise. A
    non-dict value (per-sample or ``collapse_single``) is returned unchanged.
    """
    if not isinstance(value, dict):
        return value
    if name in value:
        return value[name]
    if not value:
        return None
    if len(value) == 1:
        return next(iter(value.values()))
    raise ValueError(
        f"Modality '{name}' matched {len(value)} files for this sample "
        f"({', '.join(sorted(map(str, value)))}); expected one calibration file"
    )


def extract_intrinsics(sample: dict) -> np.ndarray | None:
    """Extract the 3x3 intrinsics matrix from a sample dict.

    Per-sample intrinsics are stored directly as ``sample["intrinsics"]``.
    Hierarchical modalities are stored as ``sample[name][file_id]``: the file
    named ``intrinsics`` or else the only matched file is used (see
    :func:`single_calibration`).
    """
    raw = single_calibration(sample.get("intrinsics"))
    if raw is None:
        return None
    return _to_numpy(raw).astype(np.float32)


def planar_to_radial_depth(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Convert planar (z-buffer) depth to radial (Euclidean) depth.

    For each pixel ``(u, v)`` the radial distance equals
    ``depth[v, u] * sqrt(((u - cx)/fx)**2 + ((v - cy)/fy)**2 + 1)``.

    Args:
        depth: ``(H, W)`` float32 planar depth in metres.
        K: ``(3, 3)`` float32 camera intrinsics matrix.

    Returns:
        ``(H, W)`` float32 radial depth in metres.
    """
    H, W = depth.shape
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    u_grid, v_grid = np.meshgrid(u, v)
    factor = np.sqrt(((u_grid - cx) / fx) ** 2 + ((v_grid - cy) / fy) ** 2 + 1.0)
    return depth * factor


def planar_to_radial_depth_torch(
    depth: "torch.Tensor", K: "torch.Tensor",
) -> "torch.Tensor":
    """Torch version of :func:`planar_to_radial_depth`.

    Args:
        depth: ``(H, W)`` or ``(B, H, W)`` float32 planar depth.
        K: ``(3, 3)`` float32 intrinsics matrix (single, shared across batch).

    Returns:
        Same shape as *depth*, converted to radial depth.
    """
    if depth.ndim == 3:
        H, W = depth.shape[1], depth.shape[2]
    else:
        H, W = depth.shape
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    u = torch.arange(W, device=depth.device, dtype=depth.dtype)
    v = torch.arange(H, device=depth.device, dtype=depth.dtype)
    v_grid, u_grid = torch.meshgrid(v, u, indexing="ij")
    factor = torch.sqrt(((u_grid - cx) / fx) ** 2 + ((v_grid - cy) / fy) ** 2 + 1.0)
    return depth * factor
