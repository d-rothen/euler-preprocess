"""Transforms as a third-party package would ship them, for entry-point tests."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, ClassVar

from euler_preprocess.common.io import load_json
from euler_preprocess.common.normalize import normalize_depth
from euler_preprocess.common.output import OutputSlotSpec
from euler_preprocess.common.transform import Transform


class ScaleDepthTransform(Transform):
    """Multiply depth by a constant factor.

    Config JSON keys: ``factor`` (number, default 2).
    """

    REQUIRED_MODALITIES: ClassVar[set[str]] = {"depth"}
    SOURCE_MODALITY: ClassVar[str] = "depth"
    OUTPUT_SLOT: ClassVar[str] = "depth"
    OUTPUT_INDEX_META_OVERRIDES: ClassVar[dict[str, object]] = {"scaled": True}
    CONFIG_HINTS: ClassVar[dict[str, dict[str, Any]]] = {
        "factor": {"type": "number", "default": 2.0, "description": "Depth multiplier."},
    }

    def __init__(self, config_path: str, out_path: str, output_backend: Any = None) -> None:
        self.factor = float(load_json(Path(config_path)).get("factor", 2.0))
        self.output_backend = output_backend
        self.out_path = Path(out_path)

    def run(self, samples: Iterable[dict]) -> list[Path]:
        paths = [
            self.output_backend.write(
                sample,
                normalize_depth(sample["depth"]) * self.factor,
                default_path=self.out_path / f"{sample['id']}.npy",
            )
            for sample in samples
        ]
        self.output_backend.finalize()
        return paths


class NotFog(ScaleDepthTransform):
    """Claims a built-in name."""


class NotATransform:
    REQUIRED_MODALITIES = {"depth"}


class Unfinished(Transform):
    """Does not implement run."""

    REQUIRED_MODALITIES: ClassVar[set[str]] = {"depth"}
    SOURCE_MODALITY: ClassVar[str] = "depth"

    def __init__(self, config_path: str, out_path: str) -> None:
        pass


class ForeignSource(ScaleDepthTransform):
    """Mirrors a modality it does not take."""

    SOURCE_MODALITY: ClassVar[str] = "rgb"


class AuxiliaryWithoutBackends(ScaleDepthTransform):
    """Declares an auxiliary slot but cannot receive its backend."""

    OUTPUT_SLOTS: ClassVar[tuple[str, ...]] = ("depth", "confidence")
    OUTPUT_SLOT_SPECS: ClassVar[dict[str, OutputSlotSpec]] = {
        "confidence": OutputSlotSpec(
            source_modality="depth",
            writer=lambda target, value, meta: None,
            index_overlay={"type": "map_2d"},
        )
    }
