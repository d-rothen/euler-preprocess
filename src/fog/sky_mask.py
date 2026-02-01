"""Sky mask extraction from segmentation images.

Provides a transform function compatible with euler-loading's
MultiModalDataset that derives a boolean sky_mask from a segmentation
modality.

Handles both CPU-format (numpy HWC) and GPU-format (torch CHW) inputs.
"""

import numpy as np

try:
    import torch
except ImportError:
    torch = None


def _seg_to_numpy_hwc(seg) -> np.ndarray:
    """Convert a segmentation array/tensor to numpy (H, W, 3)."""
    if torch is not None and torch.is_tensor(seg):
        seg = seg.detach().cpu().numpy()
    seg = np.asarray(seg)
    # CHW → HWC when first dim is small (channel) and spatial dims are large
    if seg.ndim == 3 and seg.shape[0] in (1, 3, 4) and seg.shape[1] > 4 and seg.shape[2] > 4:
        seg = np.transpose(seg, (1, 2, 0))
    return seg


def make_sky_mask(segmentation: np.ndarray, sky_color: list[int]) -> np.ndarray:
    """Create a boolean sky mask from a segmentation image.

    Args:
        segmentation: Segmentation image, shape (H, W, 3).
        sky_color: RGB color identifying sky pixels, e.g. [90, 200, 255].

    Returns:
        Boolean mask, shape (H, W), True where sky.
    """
    return (
        (segmentation[:, :, 0] == sky_color[0])
        & (segmentation[:, :, 1] == sky_color[1])
        & (segmentation[:, :, 2] == sky_color[2])
    )


def sky_mask_transform(sky_color: list[int]):
    """Return an euler-loading transform that adds ``sky_mask`` to the sample.

    The transform reads ``sample["classSegmentation"]``, computes the boolean
    sky mask, and stores it under ``sample["sky_mask"]``.

    Args:
        sky_color: RGB color identifying sky pixels.

    Returns:
        A callable ``(dict) -> dict`` suitable for the *transforms* list
        of ``MultiModalDataset``.
    """

    def _transform(sample: dict) -> dict:
        sample["sky_mask"] = make_sky_mask(
            _seg_to_numpy_hwc(sample["classSegmentation"]), sky_color
        )
        return sample

    return _transform
