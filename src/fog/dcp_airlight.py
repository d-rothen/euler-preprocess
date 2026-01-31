import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class DCPAirlight:
    """Estimate atmospheric light using the Dark Channel Prior."""

    def __init__(self, patch_size: int = 15, top_percent: float = 0.001) -> None:
        patch_size = int(patch_size)
        if patch_size < 1 or patch_size % 2 == 0:
            raise ValueError("patch_size must be a positive odd integer")
        top_percent = float(top_percent)
        if not 0.0 < top_percent <= 1.0:
            raise ValueError("top_percent must be in (0, 1]")
        self.patch_size = patch_size
        self.top_percent = top_percent

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        return self.compute(rgb)

    def compute(self, rgb: np.ndarray) -> np.ndarray:
        image = self._prepare_rgb(rgb)
        dark_channel = self._dark_channel(image)

        num_pixels = dark_channel.size
        if num_pixels == 0:
            raise ValueError("rgb image is empty")

        num_select = max(1, int(math.ceil(num_pixels * self.top_percent)))
        num_select = min(num_select, num_pixels)

        flat_dark = dark_channel.reshape(-1)
        if num_select == num_pixels:
            candidate_idx = np.arange(num_pixels)
        else:
            candidate_idx = np.argpartition(flat_dark, -num_select)[-num_select:]

        intensity = image.sum(axis=2).reshape(-1)
        best_idx = candidate_idx[np.argmax(intensity[candidate_idx])]
        airlight = image.reshape(-1, 3)[best_idx]

        return airlight.astype(np.float32)

    def _prepare_rgb(self, rgb: np.ndarray) -> np.ndarray:
        image = np.asarray(rgb)
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        elif image.ndim == 3 and image.shape[-1] == 4:
            image = image[..., :3]

        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("rgb image must have shape (H, W, 3)")

        image = image.astype(np.float32)
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        return image

    def _dark_channel(self, rgb: np.ndarray) -> np.ndarray:
        min_channel = np.min(rgb, axis=2)
        if self.patch_size == 1:
            return min_channel

        pad = self.patch_size // 2
        padded = np.pad(min_channel, ((pad, pad), (pad, pad)), mode="edge")
        windows = sliding_window_view(padded, (self.patch_size, self.patch_size))
        return np.min(windows, axis=(-1, -2))
