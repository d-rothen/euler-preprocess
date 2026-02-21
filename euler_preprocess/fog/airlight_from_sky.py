from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("foggify")

# Default fallback when no sky pixels are available (white, standard
# assumption in the Koschmieder atmospheric scattering model).
DEFAULT_AIRLIGHT_FALLBACK = np.array([1.0, 1.0, 1.0], dtype=np.float32)


class AirlightFromSky:
    def __init__(self, sky_depth_threshold: float) -> None:
        self.sky_depth_threshold = sky_depth_threshold

    def estimate_airlight(
        self,
        image: np.ndarray,
        sky_mask: np.ndarray,
        sample_id: str | None = None,
    ) -> np.ndarray:
        n_sky = int(sky_mask.sum())
        if n_sky == 0:
            id_str = f" (sample {sample_id})" if sample_id else ""
            logger.warning(
                "No sky pixels in segmentation mask%s; "
                "using default airlight fallback %s",
                id_str,
                DEFAULT_AIRLIGHT_FALLBACK.tolist(),
            )
            return DEFAULT_AIRLIGHT_FALLBACK.copy()

        airlight_pixels = image[sky_mask]
        airlight = np.mean(airlight_pixels, axis=0)

        if not np.all(np.isfinite(airlight)):
            id_str = f" (sample {sample_id})" if sample_id else ""
            logger.warning(
                "Airlight estimated from sky pixels contains non-finite values "
                "(%s)%s; using default airlight fallback %s",
                airlight,
                id_str,
                DEFAULT_AIRLIGHT_FALLBACK.tolist(),
            )
            return DEFAULT_AIRLIGHT_FALLBACK.copy()

        return airlight
