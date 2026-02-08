"""Dataset builder for real-drive-sim.

Provides ``build_dataset`` which constructs a ``MultiModalDataset`` using
euler-loading's real_drive_sim loaders and a sky-mask transform suited for
single-channel class-ID segmentation.

Dataset config keys
-------------------
modalities.rgb              Path to RGB images (RGBA PNGs).
modalities.depth            Path to depth maps (``.npz``, metres).
modalities.classSegmentation  Path to semantic segmentation PNGs.
modalities.intrinsics       Path to calibration JSONs (hierarchical, per-scene).
sky_class_id                Integer class ID that represents sky (default 29).
"""

import json

import numpy as np

from euler_loading import Modality, MultiModalDataset

try:
    import torch
except ImportError:
    torch = None

SKY_CLASS_ID = 29


# ---------------------------------------------------------------------------
# CPU intrinsics loader (numpy-only, no torch dependency)
# ---------------------------------------------------------------------------

def _read_intrinsics_numpy(path: str, meta=None) -> np.ndarray:
    """Load the CS_FRONT intrinsics from a Real Drive Sim calibration JSON.

    Returns a ``(3, 3)`` float32 camera-intrinsics matrix.
    """
    with open(path) as f:
        data = json.load(f)
    intr = dict(zip(data["names"], data["intrinsics"]))["CS_FRONT"]
    fx, fy = intr["fx"], intr["fy"]
    cx, cy = intr["cx"], intr["cy"]
    s = intr["skew"]
    return np.array(
        [[fx, s, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Sky-mask transform (single-channel segmentation)
# ---------------------------------------------------------------------------

def _to_numpy_2d(seg) -> np.ndarray:
    """Convert a segmentation array or tensor to numpy ``(H, W)``."""
    if torch is not None and torch.is_tensor(seg):
        seg = seg.detach().cpu().numpy()
    seg = np.asarray(seg)
    if seg.ndim == 3:
        seg = seg[0]  # (1, H, W) -> (H, W)
    return seg


def _sky_mask_transform(sky_class_id: int):
    """Return a transform that adds *sky_mask* from single-channel class IDs."""

    def _transform(sample: dict) -> dict:
        seg = _to_numpy_2d(sample["classSegmentation"])
        sample["sky_mask"] = seg == sky_class_id
        return sample

    return _transform


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_dataset(config: dict, use_gpu: bool) -> MultiModalDataset:
    """Build a :class:`MultiModalDataset` for the real-drive-sim dataset."""
    if use_gpu:
        from euler_loading.loaders.gpu import real_drive_sim as loaders
        intrinsics_loader = loaders.read_intrinsics
    else:
        from euler_loading.loaders.cpu import real_drive_sim as loaders
        intrinsics_loader = _read_intrinsics_numpy

    modality_paths = config["modalities"]
    sky_class_id = config.get("sky_class_id", SKY_CLASS_ID)

    return MultiModalDataset(
        modalities={
            "rgb": Modality(modality_paths["rgb"], loader=loaders.rgb),
            "depth": Modality(modality_paths["depth"], loader=loaders.depth),
            "classSegmentation": Modality(
                modality_paths["classSegmentation"],
                loader=loaders.class_segmentation,
            ),
        },
        hierarchical_modalities={
            "intrinsics": Modality(
                modality_paths["intrinsics"], loader=intrinsics_loader,
            ),
        },
        transforms=[_sky_mask_transform(sky_class_id)],
    )
