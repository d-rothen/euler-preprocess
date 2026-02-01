import numpy as np


class AirlightFromSky:
    def __init__(self, sky_depth_threshold: float) -> None:
        self.sky_depth_threshold = sky_depth_threshold

    def estimate_airlight(self, image: np.ndarray, sky_mask: np.ndarray) -> np.ndarray:
        # Ensure the image is in the correct format
        airlight_pixels = image[sky_mask]
        airlight = np.mean(airlight_pixels, axis=0)

        return airlight
