"""Crop configuration: parsing and crop-box resolution.

Pure functions, no I/O. A crop is described relative to the reference raster
(``rgb`` by default) in one of two forms:

- ``remove``: what to cut off each edge, ``{"top", "bottom", "left", "right"}``;
- ``keep``: the kept ``{"height", "width"}``, placed by ``anchor``.

Values are fractions of the reference size (``unit: "fraction"``, the default)
or whole pixels (``unit: "pixel"``). Fractions become pixels once per distinct
reference size, rounding half up (``floor(f * size + 0.5)``).

An optional ``resize`` then scales the cropped image to a new size. The whole
pixel-grid change is ``S @ C`` (crop ``C``, then ``align_corners=False`` resize
``S``); camera matrices become ``K' = S @ C @ K``.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

EDGES = ("top", "bottom", "left", "right")
KEEP_SIDES = ("height", "width")
UNITS = ("fraction", "pixel")
ANCHORS = (
    "center",
    "top",
    "bottom",
    "left",
    "right",
    "top_left",
    "top_right",
    "bottom_left",
    "bottom_right",
)
POINT_CLOUD_MODES = ("keep", "prune_to_crop")
VERIFY_MODES = ("all", "none")
ROLE_OVERRIDES = ("raster", "copy")
INTERPOLATIONS = ("nearest", "bilinear", "bicubic")

_CONFIG_KEYS = {"reference", "crop", "resize", "point_clouds", "projection", "verify", "roles"}
_CROP_KEYS = {"remove", "keep", "unit", "anchor", "multiple_of"}
_RESIZE_KEYS = {"size", "scale", "interpolation"}
_PROJECTION_KEYS = {"intrinsics", "extrinsics"}


@dataclass(frozen=True)
class CropBox:
    """A crop in reference pixels: rows ``top:top+height``, columns ``left:left+width``.

    ``output_height`` x ``output_width`` is the size after the optional resize;
    without one it equals ``height`` x ``width``.
    """

    top: int
    left: int
    height: int
    width: int
    source_height: int
    source_width: int
    output_height: int = 0
    output_width: int = 0

    def __post_init__(self) -> None:
        if not self.output_height:
            object.__setattr__(self, "output_height", self.height)
        if not self.output_width:
            object.__setattr__(self, "output_width", self.width)

    @property
    def resized(self) -> bool:
        return (self.output_height, self.output_width) != (self.height, self.width)

    @property
    def pixel_transform(self) -> tuple[tuple[float, float, float], ...]:
        """``S @ C``: source pixel coordinates to output pixel coordinates (pixel centres at integers)."""
        sy = self.output_height / self.height
        sx = self.output_width / self.width
        return (
            (sx, 0.0, sx * -self.left + (sx - 1.0) / 2.0),
            (0.0, sy, sy * -self.top + (sy - 1.0) / 2.0),
            (0.0, 0.0, 1.0),
        )

    @property
    def bottom(self) -> int:
        """Rows removed below the box."""
        return self.source_height - self.top - self.height

    @property
    def right(self) -> int:
        """Columns removed right of the box."""
        return self.source_width - self.left - self.width

    def to_dict(self) -> dict[str, int]:
        return {
            "top": self.top,
            "left": self.left,
            "height": self.height,
            "width": self.width,
            "source_height": self.source_height,
            "source_width": self.source_width,
            "output_height": self.output_height,
            "output_width": self.output_width,
        }

    def describe(self) -> str:
        text = (
            f"{self.source_height}x{self.source_width} -> {self.height}x{self.width} "
            f"(top {self.top}, bottom {self.bottom}, left {self.left}, right {self.right})"
        )
        if self.resized:
            text += f", resized to {self.output_height}x{self.output_width}"
        return text


@dataclass(frozen=True)
class CropSpec:
    """How to crop, independent of the reference size."""

    mode: str
    values: dict[str, float]
    unit: str = "fraction"
    anchor: str = "center"
    multiple_of: int | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {self.mode: dict(self.values), "unit": self.unit}
        if self.mode == "keep":
            result["anchor"] = self.anchor
        if self.multiple_of is not None:
            result["multiple_of"] = self.multiple_of
        return result

    def resolve(self, height: int, width: int) -> CropBox:
        """Return the crop box for a reference raster of ``height`` x ``width`` pixels."""
        if height <= 0 or width <= 0:
            raise ValueError(f"Reference size must be positive, got {height}x{width}")
        if self.mode == "remove":
            top, kept_h, _ = self._remove_axis(height, "top", "bottom")
            left, kept_w, _ = self._remove_axis(width, "left", "right")
            return CropBox(top, left, kept_h, kept_w, height, width)

        kept_h = self._trim(self._keep_size(height, "height"), "height")
        kept_w = self._trim(self._keep_size(width, "width"), "width")
        vertical, horizontal = _anchor_sides(self.anchor)
        top = {"start": 0, "end": height - kept_h, "center": (height - kept_h) // 2}[vertical]
        left = {"start": 0, "end": width - kept_w, "center": (width - kept_w) // 2}[horizontal]
        return CropBox(top, left, kept_h, kept_w, height, width)

    def _pixels(self, value: float, size: int) -> int:
        if self.unit == "pixel":
            return int(value)
        return math.floor(value * size + 0.5)

    def _remove_axis(self, size: int, start: str, end: str) -> tuple[int, int, int]:
        """Return ``(offset, kept, removed_at_end)`` along one axis."""
        before = self._pixels(self.values.get(start, 0), size)
        after = self._pixels(self.values.get(end, 0), size)
        kept = size - before - after
        if kept < 1:
            raise ValueError(
                f"Removing {before} px ({start}) and {after} px ({end}) leaves no "
                f"pixels of a {size} px reference"
            )
        trimmed = self._trim(kept, "height" if start == "top" else "width")
        extra = kept - trimmed
        before += extra // 2
        return before, trimmed, after + extra - extra // 2

    def _keep_size(self, size: int, side: str) -> int:
        value = self.values.get(side)
        if value is None:
            return size
        kept = self._pixels(value, size)
        if kept < 1:
            raise ValueError(f"Keeping {value} of a {size} px reference {side} leaves no pixels")
        if kept > size:
            raise ValueError(f"Cannot keep {kept} px of a {size} px reference {side}")
        return kept

    def _trim(self, kept: int, side: str) -> int:
        if self.multiple_of is None:
            return kept
        trimmed = kept - kept % self.multiple_of
        if trimmed < 1:
            raise ValueError(
                f"multiple_of {self.multiple_of} is larger than the kept {side} of {kept} px"
            )
        return trimmed


@dataclass(frozen=True)
class ResizeSpec:
    """Scale the cropped image to a new size: ``size`` in pixels or a ``scale`` factor."""

    size: tuple[int, int] | None = None
    scale: tuple[float, float] | None = None
    interpolation: dict[str, str] | None = None

    def resolve(self, height: int, width: int) -> tuple[int, int]:
        """Return the output ``(height, width)`` for a crop of ``height`` x ``width``."""
        if self.size is not None:
            return self.size
        sy, sx = self.scale or (1.0, 1.0)
        out = (max(1, math.floor(sy * height + 0.5)), max(1, math.floor(sx * width + 0.5)))
        return out

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self.size is not None:
            result["size"] = list(self.size)
        if self.scale is not None:
            result["scale"] = list(self.scale)
        if self.interpolation:
            result["interpolation"] = dict(self.interpolation)
        return result


@dataclass(frozen=True)
class CropConfig:
    """Parsed ``crop`` transform config."""

    crop: CropSpec
    reference: str = "rgb"
    resize: ResizeSpec | None = None
    point_clouds: str = "keep"
    projection: dict[str, str] | None = None
    verify: str | int = "all"
    roles: dict[str, str] | None = None

    def role_override(self, name: str) -> str | None:
        return (self.roles or {}).get(name)

    def interpolation_override(self, name: str) -> str | None:
        return ((self.resize.interpolation if self.resize else None) or {}).get(name)

    def geometry(self, height: int, width: int) -> CropBox:
        """Return the crop box, with its output size, for a reference of ``height`` x ``width``."""
        box = self.crop.resolve(height, width)
        if self.resize is None:
            return box
        out_h, out_w = self.resize.resolve(box.height, box.width)
        return CropBox(
            box.top, box.left, box.height, box.width,
            box.source_height, box.source_width, out_h, out_w,
        )

    def to_dict(self) -> dict[str, Any]:
        """The geometry part of the config, as recorded in output heads."""
        result: dict[str, Any] = {"crop": self.crop.to_dict()}
        if self.resize is not None:
            result["resize"] = self.resize.to_dict()
        result["point_clouds"] = self.point_clouds
        return result


def spatial_preprocessing(box: CropBox, *, reference: str = "rgb") -> dict[str, Any]:
    """Return the ``preprocessing`` block for ``euler-preprocess spatial`` that reproduces *box*.

    The crop is an explicit ``size`` and ``offset`` (euler-loading ``Crop``),
    followed by a ``resize`` when the box is resized.
    """
    operations: list[dict[str, Any]] = [
        {"type": "crop", "size": [box.height, box.width], "offset": [box.top, box.left]}
    ]
    if box.resized:
        operations.append({"type": "resize", "size": [box.output_height, box.output_width]})
    return {"operations": operations, "reference_field": reference}


def _anchor_sides(anchor: str) -> tuple[str, str]:
    vertical = "start" if "top" in anchor else "end" if "bottom" in anchor else "center"
    horizontal = "start" if "left" in anchor else "end" if "right" in anchor else "center"
    return vertical, horizontal


def _is_number(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(value)


def _parse_value(value: Any, *, key: str, unit: str, mode: str) -> float:
    if not _is_number(value):
        raise ValueError(f"crop.{mode}.{key} must be a number, got {value!r}")
    if unit == "pixel":
        if int(value) != value:
            raise ValueError(f"crop.{mode}.{key} must be a whole number of pixels, got {value!r}")
        if value < 0 or (mode == "keep" and value < 1):
            raise ValueError(f"crop.{mode}.{key} must be {'>= 1' if mode == 'keep' else '>= 0'} px, got {value!r}")
        return int(value)
    if mode == "remove" and not 0 <= value < 1:
        raise ValueError(
            f"crop.remove.{key} must be a fraction in [0, 1), got {value!r}; "
            "use unit 'pixel' for pixel counts (10 % is 0.10)"
        )
    if mode == "keep" and not 0 < value <= 1:
        raise ValueError(
            f"crop.keep.{key} must be a fraction in (0, 1], got {value!r}; "
            "use unit 'pixel' for pixel counts (75 % is 0.75)"
        )
    return float(value)


def parse_crop_spec(raw: Any) -> CropSpec:
    if not isinstance(raw, Mapping):
        raise ValueError("crop must be an object with 'remove' or 'keep'")
    unknown = sorted(set(raw) - _CROP_KEYS)
    if unknown:
        raise ValueError(f"crop contains unknown keys: {', '.join(unknown)}")
    modes = [mode for mode in ("remove", "keep") if mode in raw]
    if len(modes) != 1:
        raise ValueError("crop must contain exactly one of 'remove' or 'keep'")
    mode = modes[0]

    unit = raw.get("unit", "fraction")
    if unit not in UNITS:
        raise ValueError(f"crop.unit must be one of {', '.join(UNITS)}, got {unit!r}")

    sides = EDGES if mode == "remove" else KEEP_SIDES
    body = raw[mode]
    if not isinstance(body, Mapping) or not body:
        raise ValueError(f"crop.{mode} must be a non-empty object with keys among {', '.join(sides)}")
    unknown = sorted(set(body) - set(sides))
    if unknown:
        raise ValueError(f"crop.{mode} contains unknown keys: {', '.join(unknown)}")
    values = {key: _parse_value(value, key=key, unit=unit, mode=mode) for key, value in body.items()}

    if mode == "remove" and unit == "fraction":
        for start, end in (("top", "bottom"), ("left", "right")):
            if values.get(start, 0) + values.get(end, 0) >= 1:
                raise ValueError(f"crop.remove.{start} + crop.remove.{end} must be below 1")

    anchor = raw.get("anchor", "center")
    if "anchor" in raw and mode != "keep":
        raise ValueError("crop.anchor only applies to 'keep'")
    if anchor not in ANCHORS:
        raise ValueError(f"crop.anchor must be one of {', '.join(ANCHORS)}, got {anchor!r}")

    multiple_of = raw.get("multiple_of")
    if multiple_of is not None and (
        isinstance(multiple_of, bool) or not isinstance(multiple_of, int) or multiple_of < 1
    ):
        raise ValueError(f"crop.multiple_of must be a positive integer, got {multiple_of!r}")

    return CropSpec(mode=mode, values=values, unit=unit, anchor=anchor, multiple_of=multiple_of)


def parse_crop_config(raw: Any) -> CropConfig:
    """Parse and validate the ``crop`` transform config."""
    if not isinstance(raw, Mapping):
        raise ValueError("The crop config must be a JSON object")
    if "preprocessing" in raw:
        raise ValueError("The crop config must not contain 'preprocessing'; it would crop twice")
    unknown = sorted(set(raw) - _CONFIG_KEYS)
    if unknown:
        raise ValueError(f"Unknown crop config keys: {', '.join(unknown)}")
    if "crop" not in raw:
        raise ValueError("The crop config needs a 'crop' entry")

    reference = raw.get("reference", "rgb")
    if not isinstance(reference, str) or not reference:
        raise ValueError("reference must name a modality")

    point_clouds = raw.get("point_clouds", "keep")
    if point_clouds not in POINT_CLOUD_MODES:
        raise ValueError(
            f"point_clouds must be one of {', '.join(POINT_CLOUD_MODES)}, got {point_clouds!r}"
        )

    verify = raw.get("verify", "all")
    if not (
        verify in VERIFY_MODES
        or (isinstance(verify, int) and not isinstance(verify, bool) and verify > 0)
    ):
        raise ValueError(f"verify must be 'all', 'none' or a positive sample count, got {verify!r}")

    roles = raw.get("roles") or {}
    if not isinstance(roles, Mapping):
        raise ValueError("roles must map modality names to 'raster' or 'copy'")
    for name, role in roles.items():
        if role not in ROLE_OVERRIDES:
            raise ValueError(f"roles.{name} must be 'raster' or 'copy', got {role!r}")

    projection = raw.get("projection")
    if projection is not None:
        if not isinstance(projection, Mapping):
            raise ValueError("projection must name the 'intrinsics' and/or 'extrinsics' modalities")
        unknown = sorted(set(projection) - _PROJECTION_KEYS)
        if unknown:
            raise ValueError(f"projection contains unknown keys: {', '.join(unknown)}")
        for key, name in projection.items():
            if not isinstance(name, str) or not name:
                raise ValueError(f"projection.{key} must name a modality")
        if point_clouds != "prune_to_crop":
            raise ValueError("projection only applies to point_clouds 'prune_to_crop'")

    return CropConfig(
        crop=parse_crop_spec(raw["crop"]),
        reference=reference,
        resize=parse_resize_spec(raw["resize"]) if raw.get("resize") is not None else None,
        point_clouds=point_clouds,
        projection=dict(projection) if projection else None,
        verify=verify,
        roles=dict(roles),
    )


def _positive_pair(value: Any, *, key: str, kind: type) -> tuple[Any, Any]:
    values = list(value) if isinstance(value, (list, tuple)) else [value, value]
    if len(values) != 2 or not all(_is_number(v) and v > 0 for v in values):
        raise ValueError(f"resize.{key} must be a positive number or a pair [height, width]")
    if kind is int:
        if any(int(v) != v for v in values):
            raise ValueError(f"resize.{key} must be whole pixels")
        return int(values[0]), int(values[1])
    return float(values[0]), float(values[1])


def parse_resize_spec(raw: Any) -> ResizeSpec:
    if not isinstance(raw, Mapping):
        raise ValueError("resize must be an object with 'size' or 'scale'")
    unknown = sorted(set(raw) - _RESIZE_KEYS)
    if unknown:
        raise ValueError(f"resize contains unknown keys: {', '.join(unknown)}")
    if ("size" in raw) == ("scale" in raw):
        raise ValueError("resize must contain exactly one of 'size' or 'scale'")
    size = _positive_pair(raw["size"], key="size", kind=int) if "size" in raw else None
    if "size" in raw and not isinstance(raw["size"], (list, tuple)):
        raise ValueError("resize.size must be [height, width]")
    scale = _positive_pair(raw["scale"], key="scale", kind=float) if "scale" in raw else None

    interpolation = raw.get("interpolation") or {}
    if not isinstance(interpolation, Mapping):
        raise ValueError("resize.interpolation must map modality names to an interpolation")
    for name, mode in interpolation.items():
        if mode not in INTERPOLATIONS:
            raise ValueError(
                f"resize.interpolation.{name} must be one of {', '.join(INTERPOLATIONS)}, got {mode!r}"
            )
    return ResizeSpec(size=size, scale=scale, interpolation=dict(interpolation) or None)
