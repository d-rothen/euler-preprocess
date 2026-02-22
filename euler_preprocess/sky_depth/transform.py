from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar

import numpy as np

from euler_preprocess.common.io import load_json, save_depth_npy
from euler_preprocess.common.logging import get_logger, progress_bar
from euler_preprocess.common.normalize import normalize_depth, normalize_sky_mask
from euler_preprocess.common.transform import Transform


class SkyDepthTransform(Transform):
    """Override depth values in sky regions with a constant.

    Config JSON keys:
        ``sky_depth_value`` (float): Depth value to assign to sky pixels.
            Defaults to ``1000.0``.

    Each output sample is saved as a ``.npy`` float32 depth map.
    """

    REQUIRED_MODALITIES: ClassVar[set[str]] = {"depth", "semantic_segmentation"}

    def __init__(self, config_path: str, out_path: str) -> None:
        self.config_path = Path(config_path)
        self.out_path = Path(out_path)

        self.config = load_json(self.config_path)
        self.sky_depth_value = float(self.config.get("sky_depth_value", 1000.0))
        self.logger = get_logger()
        self.logger.info(
            "SkyDepthTransform: sky_depth_value=%s  output=%s",
            self.sky_depth_value,
            self.out_path,
        )

    def run(self, samples: Iterable[dict]) -> list[Path]:
        total = len(samples)  # type: ignore[arg-type]
        saved_paths: list[Path] = []

        self.out_path.mkdir(parents=True, exist_ok=True)

        with progress_bar(total, "sky-depth", self.logger) as bar:
            for sample in samples:
                depth = normalize_depth(sample["depth"])
                sky_mask = normalize_sky_mask(sample["semantic_segmentation"])
                depth[sky_mask] = self.sky_depth_value

                output_path = self._build_output_path(
                    sample["id"], full_id=sample.get("full_id"),
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                save_depth_npy(output_path, depth)
                saved_paths.append(output_path)

                if bar is not None:
                    bar.update(1)

        return saved_paths

    def _build_output_path(
        self, sample_id: str, full_id: str | None = None,
    ) -> Path:
        base = self.out_path
        if full_id:
            parts = [p for p in full_id.split("/") if p]
            if len(parts) > 1:
                base = base.joinpath(*parts[:-1])
        return base / f"{sample_id}.npy"
