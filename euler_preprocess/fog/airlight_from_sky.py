from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("euler-preprocess.fog")

# Default fallback when no sky pixels are available (white, standard
# assumption in the Koschmieder atmospheric scattering model).
DEFAULT_AIRLIGHT_FALLBACK = np.array([1.0, 1.0, 1.0], dtype=np.float32)


class NoSkyWarning:
    """Report the white-airlight fallback for samples without sky pixels.

    The first sample is logged as a warning, every later one at debug level
    only, so a run over an indoor dataset (no sky anywhere) logs one line
    instead of one per sample and fog variant. :attr:`count` holds the number
    of fallbacks so far.
    """

    def __init__(self, log: logging.Logger | None = None) -> None:
        self.log = log or logger
        self.count = 0

    def __call__(self, sample_id: str | None = None) -> None:
        self.count += 1
        id_str = f" (sample {sample_id})" if sample_id else ""
        if self.count == 1:
            self.log.warning(
                "No sky pixels in segmentation mask%s; using default airlight "
                "fallback %s. Later samples without sky pixels fall back the "
                "same way and are logged at debug level only.",
                id_str,
                DEFAULT_AIRLIGHT_FALLBACK.tolist(),
            )
        else:
            self.log.debug(
                "No sky pixels in segmentation mask%s; using default airlight "
                "fallback %s",
                id_str,
                DEFAULT_AIRLIGHT_FALLBACK.tolist(),
            )


class AirlightFromSky:
    def __init__(
        self,
        sky_depth_threshold: float,
        no_sky_warning: NoSkyWarning | None = None,
    ) -> None:
        self.sky_depth_threshold = sky_depth_threshold
        self.no_sky_warning = no_sky_warning or NoSkyWarning()

    def estimate_airlight(
        self,
        image: np.ndarray,
        sky_mask: np.ndarray,
        sample_id: str | None = None,
    ) -> np.ndarray:
        n_sky = int(sky_mask.sum())
        if n_sky == 0:
            self.no_sky_warning(sample_id)
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
