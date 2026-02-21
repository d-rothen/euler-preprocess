from __future__ import annotations

import numpy as np
from PIL import Image

try:
    import torch
except ImportError:
    torch = None


def _is_chw(arr: np.ndarray) -> bool:
    """Heuristic: 3-D array is CHW when first dim is small and spatial dims are large."""
    return (
        arr.ndim == 3
        and arr.shape[0] in (1, 3, 4)
        and arr.shape[1] > 4
        and arr.shape[2] > 4
    )


def _to_numpy(data) -> np.ndarray:
    """Convert *data* (numpy, torch tensor, or PIL Image) to a numpy array."""
    if torch is not None and torch.is_tensor(data):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def resize_depth(depth: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    height, width = target_shape
    if depth.shape == (height, width):
        return depth
    depth_img = Image.fromarray(depth.astype(np.float32), mode="F")
    depth_img = depth_img.resize((width, height), resample=Image.BILINEAR)
    return np.asarray(depth_img, dtype=np.float32)


def normalize_rgb(image) -> np.ndarray:
    image = _to_numpy(image)
    if _is_chw(image):
        image = np.transpose(image, (1, 2, 0))
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    elif image.ndim == 3 and image.shape[-1] == 4:
        image = image[..., :3]
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0
    return np.clip(image, 0.0, 1.0)


def normalize_rgb_torch(image, device: "torch.device") -> "torch.Tensor":
    if torch.is_tensor(image):
        image_t = image.to(device=device, dtype=torch.float32)
        # CHW → HWC
        if image_t.ndim == 3 and image_t.shape[0] in (1, 3, 4) and image_t.shape[2] > 4:
            image_t = image_t.permute(1, 2, 0)
        is_int = not image_t.is_floating_point()
    else:
        image_np = np.asarray(image)
        is_int = np.issubdtype(image_np.dtype, np.integer)
        image_t = torch.from_numpy(image_np).to(device)
    if image_t.ndim == 2:
        image_t = image_t.unsqueeze(-1).repeat(1, 1, 3)
    elif image_t.ndim == 3 and image_t.shape[-1] == 4:
        image_t = image_t[..., :3]
    image_t = image_t.to(torch.float32)
    if is_int:
        image_t = image_t / 255.0
    else:
        if float(image_t.max()) > 1.0:
            image_t = image_t / 255.0
    return torch.clamp(image_t, 0.0, 1.0)


def normalize_depth(
    depth, target_shape: tuple[int, int] | None = None, resize_depth_flag: bool = False
) -> np.ndarray:
    depth = _to_numpy(depth).astype(np.float32)
    # (1, H, W) → (H, W)  (GPU-loader channel-first format)
    if depth.ndim == 3 and depth.shape[0] == 1:
        depth = depth[0]
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if target_shape is not None:
        if resize_depth_flag and depth.shape != target_shape:
            depth = resize_depth(depth, target_shape)
        if depth.shape != target_shape:
            raise ValueError(
                f"Depth shape {depth.shape} does not match image shape {target_shape}"
            )
    depth[~np.isfinite(depth)] = 0.0
    depth = np.maximum(depth, 0.0)
    return depth


def normalize_sky_mask(sky_mask) -> np.ndarray:
    """Normalise a sky mask to a 2-D boolean numpy array ``(H, W)``.

    Handles torch tensors (GPU loaders) and the ``(1, H, W)`` channel-first
    layout produced by euler-loading's GPU loaders.
    """
    mask = _to_numpy(sky_mask)
    # (1, H, W) → (H, W)
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    return mask.astype(bool)
