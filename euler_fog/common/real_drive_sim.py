"""Dataset builder for real-drive-sim.

Provides ``build_dataset`` which constructs a ``MultiModalDataset`` using
euler-loading's real_drive_sim loaders and a sky-mask transform suited for
single-channel class-ID segmentation.

Dataset config keys
-------------------
modalities.rgb              Path to RGB images (RGBA PNGs).
modalities.depth            Path to depth maps (``.npz``, metres).
modalities.classSegmentation  Path to semantic segmentation PNGs.
sky_class_id                Integer class ID that represents sky (default 29).
"""

import numpy as np

from euler_loading import Modality, MultiModalDataset

try:
    import torch
except ImportError:
    torch = None

SKY_CLASS_ID = 29


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
    else:
        from euler_loading.loaders.cpu import real_drive_sim as loaders

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
        transforms=[_sky_mask_transform(sky_class_id)],
    )
