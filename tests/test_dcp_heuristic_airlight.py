from __future__ import annotations

import numpy as np
import pytest

from euler_preprocess.fog.dcp_heuristic_airlight import DCPHeuristicAirlight

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from euler_preprocess.fog.dcp_heuristic_airlight_torch import (
        DCPHeuristicAirlightTorch,
    )


def _make_scene() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    image = np.full((4, 4, 3), 0.05, dtype=np.float32)
    sky_color = np.array([0.90, 0.94, 1.00], dtype=np.float32)
    median_pixel = np.array([0.76, 0.76, 0.78], dtype=np.float32)

    image[0, 0] = sky_color
    image[0, 1] = sky_color
    image[1, 0] = [0.72, 0.72, 0.74]
    image[1, 1] = [0.74, 0.74, 0.74]
    image[1, 2] = median_pixel

    sky_mask = np.zeros((4, 4), dtype=bool)
    sky_mask[0, 0] = True
    sky_mask[0, 1] = True
    return image, sky_mask, sky_color, median_pixel


class TestDCPHeuristicAirlight:
    def test_compute_pools_bright_candidates_instead_of_single_median_pixel(self):
        image, _, sky_color, median_pixel = _make_scene()
        estimator = DCPHeuristicAirlight(patch_size=1, top_percent=0.3)

        result = estimator.compute(image)

        assert estimator._luminance(result) > estimator._luminance(median_pixel)
        assert estimator._luminance(result) < estimator._luminance(sky_color)
        assert not np.allclose(result, median_pixel)

    def test_estimate_airlight_uses_bright_sky_chromaticity_when_available(self):
        image, sky_mask, sky_color, _ = _make_scene()
        estimator = DCPHeuristicAirlight(patch_size=1, top_percent=0.3)

        result = estimator.estimate_airlight(image, sky_mask)

        np.testing.assert_allclose(result, sky_color, atol=1e-6)


@pytest.mark.skipif(torch is None, reason="torch not installed")
class TestDCPHeuristicAirlightTorch:
    def test_estimate_airlight_matches_cpu_sky_guided_result(self):
        image, sky_mask, sky_color, _ = _make_scene()
        estimator = DCPHeuristicAirlightTorch(patch_size=1, top_percent=0.3)

        result = estimator.estimate_airlight(
            torch.from_numpy(image),
            torch.from_numpy(sky_mask),
        )

        torch.testing.assert_close(
            result,
            torch.from_numpy(sky_color),
            atol=1e-6,
            rtol=0,
        )
