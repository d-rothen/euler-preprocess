"""What the crop does to each modality, decided from its loader annotation.

Every euler-loading loader carries ``@modality_meta(shape=..., type=...)``.
The role follows from that shape and never from the modality's name, so a
``sparse_depth`` point cloud or a 4x4 ``camera_extrinsics`` matrix is never
sliced like an image.

| Annotated shape | Role | Operation |
|---|---|---|
| ``HW``, ``HWC``, ``CHW``, ``1HW``, ``3HW`` | raster | Slice with the crop box |
| ``3x3`` with type ``intrinsics`` | intrinsics | ``K' = crop_intrinsics(K, top, left)`` |
| ``NC`` (point sets), ``4x4`` / ``Nx1`` (extrinsics) | copy | Copy the file unchanged |
| ``dict`` with type ``gnss`` or ``sensor_timing`` | copy | Copy the file unchanged |

Anything else needs a ``roles`` override (``"raster"`` or ``"copy"``). Other
``dict`` loaders can hold pixel-grid values (Real Drive Sim's calibration map
holds the camera matrix), so they are never copied by default.

A copied file that an ``intrinsics`` modality also indexes (MUSES
``calib.json`` holds the camera matrix and every extrinsic) is written as the
patched file, so the output carries one camera matrix.

When the crop also resizes, colour images (types ``rgb`` and ``rccb``) are
interpolated bilinearly and every other raster (depth, masks, labels) takes the
nearest source pixel, so no new depth or label values appear. Sparse-depth
rasters cannot be resized: nearest sampling would drop points at random.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

RASTER = "raster"
INTRINSICS = "intrinsics"
COPY = "copy"

#: Raster layouts and the axes holding the image height and width.
RASTER_AXES: dict[str, tuple[int, int]] = {
    "HW": (0, 1),
    "HWC": (0, 1),
    "CHW": (1, 2),
    "1HW": (1, 2),
    "3HW": (1, 2),
}
_COPY_SHAPES = {"NC", "4x4", "Nx1"}
#: ``dict`` loaders whose values never refer to the image grid (MUSES GNSS
#: records and ``meta.json`` sensor timing).
_COPY_DICT_TYPES = {"gnss", "sensor_timing"}
#: euler-loading's resize layouts for the annotated raster layouts.
_RESIZE_LAYOUTS = {"HW": "HW", "HWC": "HWC", "CHW": "CHW", "1HW": "CHW", "3HW": "CHW"}
#: Raster types that are interpolated when resized; all others use nearest.
INTERPOLATED_TYPES = {"rgb", "rccb"}


@dataclass(frozen=True)
class ModalityRole:
    """How the crop treats one modality."""

    name: str
    role: str
    hierarchical: bool
    layout: str | None = None
    shape: str | None = None
    modality_type: str | None = None

    def raster_size(self, value: Any) -> tuple[int, int]:
        """Return ``(height, width)`` of a raster value in this modality's layout."""
        h_axis, w_axis = RASTER_AXES[self.layout or ""]
        shape = tuple(value.shape)
        if len(shape) != len(self.layout or ""):
            raise ValueError(
                f"Modality '{self.name}' is annotated {self.layout} but loaded a value "
                f"of shape {shape}"
            )
        return int(shape[h_axis]), int(shape[w_axis])

    def slice(self, value: Any, top: int, left: int, height: int, width: int) -> Any:
        """Return the crop of a raster value, as a contiguous array or tensor."""
        h_axis, w_axis = RASTER_AXES[self.layout or ""]
        index = [slice(None)] * len(value.shape)
        index[h_axis] = slice(top, top + height)
        index[w_axis] = slice(left, left + width)
        cropped = value[tuple(index)]
        if hasattr(cropped, "contiguous"):
            return cropped.contiguous()
        import numpy as np

        return np.ascontiguousarray(cropped)

    @property
    def is_point_set(self) -> bool:
        return self.role == COPY and self.shape == "NC"

    def default_interpolation(self) -> str:
        return "bilinear" if self.modality_type in INTERPOLATED_TYPES else "nearest"

    def resize(self, value: Any, height: int, width: int, interpolation: str | None = None) -> Any:
        """Return the raster resized to ``height`` x ``width`` (euler-loading ``resize_raster``)."""
        from euler_loading import resize_raster

        if self.modality_type == "sparse_depth":
            raise ValueError(
                f"Modality '{self.name}' is a sparse-depth raster and cannot be resized: "
                "nearest sampling would drop points at random. Resize from point clouds instead."
            )
        return resize_raster(
            value,
            layout=_RESIZE_LAYOUTS[self.layout or ""],
            size=(height, width),
            interpolation=interpolation or self.default_interpolation(),
        )


def describe_annotation(annotation: Mapping[str, Any] | None) -> str:
    if not annotation:
        return "no @modality_meta annotation"
    return f"shape {annotation.get('shape')!r}, type {annotation.get('type')!r}"


def infer_role(
    name: str,
    annotation: Mapping[str, Any] | None,
    *,
    hierarchical: bool,
    override: str | None = None,
) -> ModalityRole:
    """Return the crop role of modality *name* from its loader annotation.

    *override* (``"raster"`` or ``"copy"``) replaces the inferred role. A
    raster still needs an annotated raster layout to know its height and
    width axes.
    """
    annotation = annotation or {}
    shape = annotation.get("shape") if isinstance(annotation.get("shape"), str) else None
    modality_type = annotation.get("type") if isinstance(annotation.get("type"), str) else None

    def role(kind: str, layout: str | None = None) -> ModalityRole:
        return ModalityRole(
            name=name,
            role=kind,
            hierarchical=hierarchical,
            layout=layout,
            shape=shape,
            modality_type=modality_type,
        )

    if override == COPY:
        return role(COPY)
    if override == RASTER:
        if shape not in RASTER_AXES:
            raise ValueError(
                f"Modality '{name}' cannot be cropped as a raster: its loader has "
                f"{describe_annotation(annotation)}, so the height and width axes are unknown"
            )
        return role(RASTER, shape)
    if override is not None:
        raise ValueError(f"Unknown role override {override!r} for modality '{name}'")

    if shape in RASTER_AXES:
        return role(RASTER, shape)
    if shape == "3x3" and modality_type == "intrinsics":
        return role(INTRINSICS)
    if shape in _COPY_SHAPES or (shape == "dict" and modality_type in _COPY_DICT_TYPES):
        return role(COPY)
    raise ValueError(
        f"Don't know how to crop modality '{name}' ({describe_annotation(annotation)}). "
        f'Set "roles": {{"{name}": "raster" | "copy"}} in the crop config.'
    )


def loader_annotation(dataset: Any, name: str) -> Mapping[str, Any] | None:
    """Return the ``@modality_meta`` annotation of the loader *dataset* uses for *name*.

    It describes the function that decodes the modality, so it is
    authoritative for the values a sample carries.
    """
    return dataset.get_loader_meta(name)
