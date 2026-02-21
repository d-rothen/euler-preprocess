from __future__ import annotations

import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class DCPHeuristicAirlight:
    """Estimate atmospheric light via Dark Channel Prior with median heuristic.

    Like :class:`DCPAirlight`, this selects the top bright pixels from the dark
    channel.  Instead of picking the single brightest pixel (by sum of
    channels), it converts candidates to grayscale (NTSC/BT.601 weights) and
    picks the pixel whose intensity equals the **median** among candidates.
    The candidate count is forced to be odd so that the median always
    corresponds to an actual pixel.
    """

    def __init__(self, patch_size: int = 15, top_percent: float = 0.001) -> None:
        patch_size = int(patch_size)
        if patch_size < 1 or patch_size % 2 == 0:
            raise ValueError("patch_size must be a positive odd integer")
        top_percent = float(top_percent)
        if not 0.0 < top_percent <= 1.0:
            raise ValueError("top_percent must be in (0, 1]")
        self.patch_size = patch_size
        self.top_percent = top_percent

    # -- public interface (matches AirlightFromSky / DCPAirlight) -----------

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        return self.compute(rgb)

    def compute(self, rgb: np.ndarray) -> np.ndarray:
        image = self._prepare_rgb(rgb)
        dark_channel = self._dark_channel(image)

        num_pixels = dark_channel.size
        if num_pixels == 0:
            raise ValueError("rgb image is empty")

        num_select = self._brightest_pixels_count(num_pixels)
        flat_dark = dark_channel.ravel()

        if num_select >= num_pixels:
            candidate_idx = np.arange(num_pixels)
        else:
            candidate_idx = np.argpartition(flat_dark, -num_select)[-num_select:]

        # NTSC/BT.601 luminance weights
        i_gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]).ravel()
        candidate_gray = i_gray[candidate_idx]

        median_intensity = np.median(candidate_gray)
        local_best_idx = np.argmin(np.abs(candidate_gray - median_intensity))
        best_idx = candidate_idx[local_best_idx]

        airlight = image.reshape(-1, 3)[best_idx]
        return airlight.astype(np.float32)

    def estimate_airlight(
        self,
        image: np.ndarray,
        sky_mask: np.ndarray,
        sample_id: str | None = None,
    ) -> np.ndarray:
        """Unified interface compatible with :class:`AirlightFromSky`.

        *sky_mask* is accepted for interface compatibility but not used;
        the dark channel prior estimates airlight from the full image.
        """
        return self.compute(image)

    # -- internals ----------------------------------------------------------

    def _brightest_pixels_count(self, num_pixels: int) -> int:
        count = int(math.floor(self.top_percent * num_pixels))
        # Force odd so that the median matches an actual pixel
        count = count + ((count + 1) % 2)
        return max(1, min(count, num_pixels))

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
