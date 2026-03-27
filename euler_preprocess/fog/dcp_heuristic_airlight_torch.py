from __future__ import annotations

import math

import torch
import torch.nn.functional as F

# NTSC / BT.601 luminance weights
_GRAY_WEIGHTS = torch.tensor([0.2989, 0.5870, 0.1140])
_BRIGHT_SKY_QUANTILE = 0.75


class DCPHeuristicAirlightTorch:
    """GPU variant of :class:`DCPHeuristicAirlight` using pure torch operations.

    Uses the Dark Channel Prior with a robust pooled heuristic:
    the brighter half of the top *top_percent* dark-channel pixels are
    averaged with luminance weights. When a sky mask is available, the
    brightest sky pixels provide the chromaticity prior and the DCP
    estimate contributes the target luminance.

    Input must be a float32 ``(H, W, 3)`` tensor already on the target
    device.  Returns a ``(3,)`` tensor on the same device.
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

    def compute(self, rgb: torch.Tensor) -> torch.Tensor:
        """Compute airlight from an ``(H, W, 3)`` float tensor."""
        image = self._prepare_rgb(rgb)
        candidate_rgb, candidate_gray = self._candidate_pixels(image)
        return self._estimate_from_candidates(candidate_rgb, candidate_gray)

    def __call__(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.compute(rgb)

    def estimate_airlight(
        self,
        image: torch.Tensor,
        sky_mask: torch.Tensor,
        sample_id: str | None = None,
    ) -> torch.Tensor:
        del sample_id
        prepared = self._prepare_rgb(image)
        candidate_rgb, candidate_gray = self._candidate_pixels(prepared)
        airlight = self._estimate_from_candidates(candidate_rgb, candidate_gray)
        sky_prior = self._estimate_sky_prior(prepared, sky_mask)
        if sky_prior is None:
            return airlight
        return self._merge_with_sky_prior(airlight, sky_prior)

    # ------------------------------------------------------------------

    def _brightest_pixels_count(self, num_pixels: int) -> int:
        count = int(math.floor(self.top_percent * num_pixels))
        # Force odd so the median corresponds to an actual pixel
        count = count + ((count + 1) % 2)
        return max(1, min(count, num_pixels))

    @staticmethod
    def _prepare_rgb(rgb: torch.Tensor) -> torch.Tensor:
        image = rgb
        if image.ndim == 2:
            image = image.unsqueeze(-1).expand(-1, -1, 3)
        elif image.ndim == 3 and image.shape[-1] == 4:
            image = image[..., :3]
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("rgb image must have shape (H, W, 3)")
        image = torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        return image.to(torch.float32)

    def _candidate_pixels(
        self,
        rgb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dark_channel = self._dark_channel(rgb)

        num_pixels = dark_channel.numel()
        if num_pixels == 0:
            raise ValueError("rgb image is empty")

        num_select = self._brightest_pixels_count(num_pixels)
        flat_dark = dark_channel.reshape(-1)
        _, candidate_idx = torch.topk(flat_dark, num_select)

        flat_rgb = rgb.reshape(-1, 3)
        candidate_rgb = flat_rgb[candidate_idx]
        weights = _GRAY_WEIGHTS.to(device=rgb.device, dtype=rgb.dtype)
        candidate_gray = (candidate_rgb * weights).sum(dim=1)
        return candidate_rgb, candidate_gray

    def _estimate_from_candidates(
        self,
        candidate_rgb: torch.Tensor,
        candidate_gray: torch.Tensor,
    ) -> torch.Tensor:
        median_intensity = torch.median(candidate_gray)
        bright_mask = candidate_gray >= median_intensity
        if not torch.any(bright_mask):
            bright_mask = torch.ones_like(candidate_gray, dtype=torch.bool)

        bright_rgb = candidate_rgb[bright_mask]
        bright_gray = candidate_gray[bright_mask]
        weights = bright_gray - bright_gray.min()
        weights = weights + torch.finfo(bright_gray.dtype).eps
        weighted_sum = (bright_rgb * weights[:, None]).sum(dim=0)
        return weighted_sum / weights.sum()

    def _estimate_sky_prior(
        self,
        rgb: torch.Tensor,
        sky_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if sky_mask is None:
            return None

        mask = sky_mask.to(device=rgb.device, dtype=torch.bool)
        if tuple(mask.shape) != tuple(rgb.shape[:2]):
            raise ValueError("sky_mask must have shape (H, W)")
        if not torch.any(mask):
            return None

        sky_pixels = rgb[mask]
        weights = _GRAY_WEIGHTS.to(device=rgb.device, dtype=rgb.dtype)
        sky_gray = (sky_pixels * weights).sum(dim=1)
        threshold = self._quantile_value(sky_gray, _BRIGHT_SKY_QUANTILE)
        bright_sky = sky_pixels[sky_gray >= threshold]
        if bright_sky.numel() == 0:
            bright_sky = sky_pixels
        return bright_sky.mean(dim=0)

    def _merge_with_sky_prior(
        self,
        airlight: torch.Tensor,
        sky_prior: torch.Tensor,
    ) -> torch.Tensor:
        sky_luminance = self._luminance(sky_prior)
        if sky_luminance <= torch.finfo(sky_prior.dtype).eps:
            return airlight

        target_luminance = max(self._luminance(airlight), sky_luminance)
        return sky_prior * (target_luminance / sky_luminance)

    @staticmethod
    def _quantile_value(values: torch.Tensor, quantile: float) -> torch.Tensor:
        sorted_values, _ = torch.sort(values)
        index = int(math.ceil(quantile * sorted_values.numel()) - 1)
        index = min(max(index, 0), sorted_values.numel() - 1)
        return sorted_values[index]

    @staticmethod
    def _luminance(rgb: torch.Tensor) -> float:
        weights = _GRAY_WEIGHTS.to(device=rgb.device, dtype=rgb.dtype)
        return float((rgb * weights).sum().item())

    def _dark_channel(self, rgb: torch.Tensor) -> torch.Tensor:
        min_channel = rgb.min(dim=-1).values
        if self.patch_size == 1:
            return min_channel
        pad = self.patch_size // 2
        # Replicate-pad first (matching numpy edge padding), then pool
        neg = (-min_channel).unsqueeze(0).unsqueeze(0)
        neg = F.pad(neg, (pad, pad, pad, pad), mode="replicate")
        pooled = F.max_pool2d(neg, kernel_size=self.patch_size, stride=1)
        return -pooled.squeeze(0).squeeze(0)
