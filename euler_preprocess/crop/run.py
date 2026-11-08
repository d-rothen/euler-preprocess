"""Crop (and optionally resize) every modality of a dataset coherently.

Per sample, the reference raster (``rgb`` by default) fixes the crop box and
the output size. The pixel-grid change is ``S @ C`` (crop ``C``, then the
``align_corners=False`` resize ``S``):

- rasters are sliced, resized if requested, and re-encoded with the dataset's
  own writer; every encoded file is decoded again and must reproduce the
  values (exactly, or to half an 8-bit step when interpolated);
- pinhole intrinsics become ``K' = S @ C @ K``, merged into a copy of their
  source file;
- extrinsics and lidar point clouds are unchanged: ``u' = S @ C @ K @ [R|t] @ X``
  only involves the camera matrix. With ``point_clouds: "prune_to_crop"`` the
  points that cannot project into the output image are dropped, using the
  projection consumers use (``rint`` of ``K' @ T @ X``, ``z > 0``).

Hierarchical files (shared calibration) are written once, after all samples,
with the box of the samples using them.

A file indexed by several modalities (MUSES ``calib.json`` is ``intrinsics``,
``camera_extrinsics`` and ``lidar_to_gnss`` at once) is patched by its
intrinsics modality, and every copied file with the same source bytes carries
the patched bytes, so the output holds one camera matrix. Matching by content
also covers a calibration file indexed as separate copies.
"""

from __future__ import annotations

import inspect
import json
import shutil
import tempfile
import zipfile
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from ds_crawler import ZipDatasetWriter
from ds_crawler.zip_utils import COMPRESSED_EXTENSIONS
from euler_loading import crop_intrinsics, resize_intrinsics

from euler_preprocess.common.intrinsics import single_calibration
from euler_preprocess.common.logging import get_logger
from euler_preprocess.common.output import (
    PipelineOutputTargetConfig,
    SourceBackedOutputBackend,
    _modality_key,
    parse_pipeline_config,
    prepare_modality_output_backend,
    write_pipeline_outputs_manifest,
)
from euler_preprocess.crop.calibration import HierarchicalFiles
from euler_preprocess.crop.files import (
    SourceFiles,
    byte_digest,
    fingerprint,
    recorded_image_size,
    to_numpy,
)
from euler_preprocess.crop.roles import (
    COPY,
    INTRINSICS,
    RASTER,
    ModalityRole,
    infer_role,
    loader_annotation,
)
from euler_preprocess.crop.spec import CropBox, CropConfig, parse_crop_config

_PROGRESS_EVERY = 500
#: Points within this many pixels outside the image are kept, so that no point
#: a consumer's rounding puts inside is ever dropped.
PRUNE_MARGIN_PX = 1e-6
_EXTRINSICS_TYPES = {"camera_extrinsics", "extrinsics"}
OUTPUT_LAYOUTS = ("per_modality", "scoped")


@dataclass(frozen=True)
class OutputLocation:
    """Where one modality's cropped dataset is written."""

    root: Path
    storage: str = "directory"
    target: PipelineOutputTargetConfig | None = None
    metadata_scope: str | None = None
    path_prefix: str | None = None


@dataclass(frozen=True)
class Expectation:
    """What a written file must reload as: a value digest, a matrix or the source bytes."""

    kind: str
    value: Any


@dataclass(frozen=True)
class PatchedFile:
    """A source file an intrinsics modality patched, reused by copies of the same file."""

    modality: str
    data: bytes
    box: CropBox


@dataclass
class CropResult:
    samples: int
    boxes: list[CropBox]
    outputs: dict[str, OutputLocation]
    roles: dict[str, ModalityRole]
    expectations: dict[tuple[str, str], Expectation] = field(default_factory=dict)
    files_written: dict[str, int] = field(default_factory=dict)
    points: dict[str, tuple[int, int]] = field(default_factory=dict)


def resolve_roles(dataset: Any, config: CropConfig) -> dict[str, ModalityRole]:
    """Return the crop role of every modality of *dataset*, regular ones first."""
    regular = list(dataset.modality_paths())
    hierarchical = list(dataset.hierarchical_modality_paths())
    names = regular + hierarchical
    unknown = sorted(set(config.roles or {}) - set(names))
    if unknown:
        raise ValueError(f"roles names modalities that are not configured: {', '.join(unknown)}")

    roles = {
        name: infer_role(
            name,
            loader_annotation(dataset, name),
            hierarchical=name in hierarchical,
            override=config.role_override(name),
        )
        for name in names
    }
    reference = config.reference
    if reference not in roles:
        raise ValueError(
            f"The reference modality '{reference}' is not in the dataset config "
            f"(configured: {', '.join(names)})"
        )
    if roles[reference].hierarchical or roles[reference].role != RASTER:
        raise ValueError(
            f"The reference modality '{reference}' must be a per-sample raster; it is a "
            f"{'hierarchical ' if roles[reference].hierarchical else ''}{roles[reference].role}"
        )
    return roles


def visible_points(
    points: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray, box: CropBox
) -> np.ndarray:
    """Return a mask of the points that can project into the output image of *box*.

    Mirrors consumer projections (e.g. dmd3c-euler ``project_sparse_depth``):
    ``X_cam = T @ X``, ``z > 0``, pixel ``rint(K' @ X_cam / z)`` inside the output
    image, with ``K' = S @ C @ K``. A margin of :data:`PRUNE_MARGIN_PX` keeps
    points exactly on a rounding boundary.
    """
    xyz = np.asarray(points, dtype=np.float64)[:, :3]
    transform = np.asarray(extrinsics, dtype=np.float64)
    camera = xyz @ transform[:3, :3].T + transform[:3, 3]
    z = camera[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        pixels = camera @ np.asarray(intrinsics, dtype=np.float64).T
        u = pixels[:, 0] / pixels[:, 2]
        v = pixels[:, 1] / pixels[:, 2]
        grid = np.asarray(box.pixel_transform, dtype=np.float64)
        out_u = grid[0, 0] * u + grid[0, 2]
        out_v = grid[1, 1] * v + grid[1, 2]
        low = -0.5 - PRUNE_MARGIN_PX
        return (
            np.isfinite(camera).all(axis=1)
            & (z > 0)
            & (out_u >= low)
            & (out_u <= box.output_width - 0.5 + PRUNE_MARGIN_PX)
            & (out_v >= low)
            & (out_v <= box.output_height - 0.5 + PRUNE_MARGIN_PX)
        )


def _accepts_attributes(loader: Any) -> bool:
    try:
        parameters = inspect.signature(loader).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == "attributes" or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _comparable(value: Any) -> bool:
    """Whether *value* is a numeric array or tensor that :func:`fingerprint` can compare."""
    if value is None:
        return False
    try:
        return to_numpy(value).dtype != object
    except Exception:
        return False


def _select(value: Any, mask: np.ndarray) -> Any:
    if hasattr(value, "detach"):
        import torch

        return value[torch.from_numpy(mask).to(value.device)].contiguous()
    return np.ascontiguousarray(np.asarray(value)[mask])


class CropRunner:
    """Crop the selected samples of *dataset* and write every modality."""

    def __init__(
        self,
        dataset: Any,
        config: CropConfig,
        outputs: Mapping[str, OutputLocation],
    ) -> None:
        self.dataset = dataset
        self.config = config
        self.logger = get_logger()
        self.roles = resolve_roles(dataset, config)
        missing = [name for name in self.roles if name not in outputs]
        if missing:
            raise ValueError(f"No output location for modalities: {', '.join(missing)}")
        self.outputs = {name: outputs[name] for name in self.roles}
        self.prune = config.point_clouds == "prune_to_crop"
        self.projection = self._projection_modalities() if self.prune else None

        writers: dict[str, Any] = {}
        for name, role in self.roles.items():
            if not (role.role in (RASTER, INTRINSICS) or (self.prune and role.is_point_set)):
                continue
            try:
                writers[name] = dataset.get_writer(name)
            except (KeyError, ValueError):
                what = "pruned points" if role.is_point_set else "cropped values"
                raise ValueError(
                    f"Modality '{name}' ({role.role}) has no euler-loading writer, so its "
                    f"{what} cannot be stored. Copy it unchanged with "
                    f'"roles": {{"{name}": "copy"}} only if it does not depend on the image grid.'
                ) from None

        self.backends: dict[str, SourceBackedOutputBackend] = {}
        try:
            for name in self.roles:
                location = self.outputs[name]
                self.backends[name] = prepare_modality_output_backend(
                    dataset,
                    name,
                    location.root,
                    storage=location.storage,
                    writer=writers.get(name),
                    metadata_scope=location.metadata_scope,
                    path_prefix=location.path_prefix,
                )
        except Exception:
            self._close_writers()
            raise

        self.sources = SourceFiles(dataset)
        self.hierarchical = HierarchicalFiles(
            dataset, [name for name, role in self.roles.items() if role.hierarchical]
        )
        self._boxes: dict[tuple[int, int], CropBox] = {}
        self._boxes_by_modality: dict[str, set[CropBox]] = defaultdict(set)
        self._seen_samples: set[str] = set()
        #: Files patched by an intrinsics modality, and files copied unchanged,
        #: by the digest of their source bytes.
        self._patched: dict[str, PatchedFile] = {}
        self._copied: dict[str, str] = {}
        self._shares_logged: set[str] = set()
        self.expectations: dict[tuple[str, str], Expectation] = {}
        self.files_written: Counter[str] = Counter()
        self.points: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    def _close_writers(self) -> None:
        """Close open zip archives after a failure (``save_index`` closes them otherwise)."""
        for backend in getattr(self, "backends", {}).values():
            if isinstance(backend.dataset_writer, ZipDatasetWriter):
                backend.dataset_writer.close()

    def _projection_modalities(self) -> tuple[str, str]:
        """Return the intrinsics and extrinsics modalities that project the point clouds."""
        point_sets = [name for name, role in self.roles.items() if role.is_point_set]
        if not point_sets:
            raise ValueError("point_clouds 'prune_to_crop' needs a point-cloud modality (shape NC)")
        hierarchical = [name for name in point_sets if self.roles[name].hierarchical]
        if hierarchical:
            raise ValueError(
                f"Cannot prune hierarchical point clouds ({', '.join(hierarchical)}): one shared "
                "file has no single camera pose"
            )

        configured = self.config.projection or {}

        def pick(key: str, candidates: list[str], what: str) -> str:
            name = configured.get(key)
            if name is not None:
                if name not in self.roles:
                    raise ValueError(f"projection.{key} names an unconfigured modality '{name}'")
                return name
            if len(candidates) != 1:
                found = ", ".join(candidates) or "none"
                raise ValueError(
                    f"point_clouds 'prune_to_crop' needs exactly one {what} modality (found: "
                    f'{found}); name it with "projection": {{"{key}": "<modality>"}}'
                )
            return candidates[0]

        intrinsics = pick(
            "intrinsics",
            [name for name, role in self.roles.items() if role.role == INTRINSICS],
            "intrinsics",
        )
        extrinsics = pick(
            "extrinsics",
            [
                name
                for name, role in self.roles.items()
                if role.shape == "4x4" and role.modality_type in _EXTRINSICS_TYPES
            ],
            "4x4 camera_extrinsics",
        )
        if self.roles[intrinsics].role != INTRINSICS:
            raise ValueError(f"projection.intrinsics '{intrinsics}' is not a 3x3 intrinsics modality")
        if self.roles[extrinsics].shape != "4x4":
            raise ValueError(f"projection.extrinsics '{extrinsics}' is not a 4x4 transform")
        self.logger.info(
            "crop: pruning %s with intrinsics '%s' and extrinsics '%s'",
            ", ".join(point_sets),
            intrinsics,
            extrinsics,
        )
        return intrinsics, extrinsics

    # -- boxes ----------------------------------------------------------------

    def box_for(self, height: int, width: int) -> CropBox:
        box = self._boxes.get((height, width))
        if box is None:
            box = self.config.geometry(height, width)
            self._boxes[(height, width)] = box
            self.logger.info("Crop for %s rasters: %s", self.config.reference, box.describe())
        return box

    # -- encoding -------------------------------------------------------------

    def _decode(self, name: str, path: Path, entry: Mapping[str, Any], attributes: Mapping[str, Any]) -> Any:
        loader = self.dataset.get_loader(name)
        meta = self.backends[name].modality_meta
        if _accepts_attributes(loader):
            return loader(str(path), meta, attributes=dict(attributes))
        return loader(str(path), meta)

    def _encode(
        self,
        name: str,
        entry: Mapping[str, Any],
        value: Any,
        attributes: Mapping[str, Any],
        *,
        tolerance: float | None,
    ) -> tuple[bytes, str]:
        """Encode *value* with the modality's writer and check that it decodes back.

        Returns the file bytes and the fingerprint of the decoded value.
        ``tolerance=None`` requires an exact round trip.
        """
        backend = self.backends[name]
        basename = Path(str(entry["path"])).name
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / basename
            backend.modality_writer(str(path), value, backend.modality_meta)
            data = path.read_bytes()
            decoded = self._decode(name, path, entry, {**(entry.get("attributes") or {}), **attributes})
        expected = to_numpy(value)
        reloaded = to_numpy(decoded)
        if reloaded.shape != expected.shape:
            raise ValueError(
                f"The '{name}' writer stored {entry['path']!r} as shape {reloaded.shape}, "
                f"not {expected.shape}"
            )
        if tolerance is None:
            ok = fingerprint(reloaded) == fingerprint(expected)
        else:
            difference = np.abs(reloaded.astype(np.float64) - expected.astype(np.float64))
            ok = bool(np.all(difference <= tolerance))
        if not ok:
            raise ValueError(
                f"The '{name}' writer does not store {entry['path']!r} faithfully: it reloads "
                f"with different values{' beyond the quantization tolerance' if tolerance else ''}"
            )
        return data, fingerprint(reloaded)

    def _tolerance(self, role: ModalityRole, interpolation: str) -> float | None:
        if interpolation == "nearest":
            return None
        annotation = loader_annotation(self.dataset, role.name) or {}
        output_range = annotation.get("output_range")
        span = (
            float(output_range[1]) - float(output_range[0])
            if isinstance(output_range, (list, tuple)) and len(output_range) == 2
            else 1.0
        )
        # Half a step of an 8-bit encoding of the value range.
        return span / 255.0 / 2.0 + 1e-6

    # -- writing --------------------------------------------------------------

    def _write_order(self) -> list[str]:
        """Modalities in config order, intrinsics first: copies of a patched file follow its patch."""
        return sorted(self.roles, key=lambda name: self.roles[name].role != INTRINSICS)

    def _remember_patch(
        self, name: str, entry: Mapping[str, Any], source: bytes, patched: bytes, box: CropBox
    ) -> None:
        key = byte_digest(source)
        other = self._patched.get(key)
        if other is not None and other.data != patched:
            raise ValueError(
                f"Intrinsics modalities '{other.modality}' and '{name}' patch the same file "
                f"({entry['path']!r}) differently. Crop it as one of them and copy it as the "
                f'other (e.g. "roles": {{"{name}": "copy"}}), so it is patched once.'
            )
        copied = self._copied.get(key)
        if copied is not None and patched != source:
            raise ValueError(
                f"Modality '{copied}' already copied {entry['path']!r} unchanged, but '{name}' "
                "patches its camera matrix; the output would hold both versions"
            )
        self._patched.setdefault(key, PatchedFile(name, patched, box))

    def _copy_bytes(
        self, name: str, entry: Mapping[str, Any], value: Any, box: CropBox
    ) -> tuple[bytes, str]:
        """Return the bytes a copied file is stored with (the source, or its patched version) and their digest."""
        data = self.sources.read(name, entry)
        key = byte_digest(data)
        patched = self._patched.get(key)
        if patched is None or patched.data == data:
            # Unpatched, or the patch changed nothing (e.g. a crop that keeps
            # the principal point, like cutting rows off the bottom).
            self._copied.setdefault(key, name)
            return data, key
        if patched.box != box:
            raise ValueError(
                f"Modalities '{patched.modality}' and '{name}' share the file {entry['path']!r} "
                f"but get different crops ({patched.box.describe()} and {box.describe()})"
            )
        if _comparable(value):
            # The patch may only change the camera matrix: what this modality
            # reads from the file must stay as it was.
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / Path(str(entry["path"])).name
                path.write_bytes(patched.data)
                reloaded = self._decode(name, path, entry, entry.get("attributes") or {})
            if fingerprint(reloaded) != fingerprint(value):
                raise ValueError(
                    f"Patching the camera matrix in {entry['path']!r} for '{patched.modality}' "
                    f"changed what '{name}' reads from it; its writer must keep every other field"
                )
        if name not in self._shares_logged:
            self._shares_logged.add(name)
            self.logger.info(
                "crop: '%s' indexes the file '%s' patches (%s); it is written patched",
                name,
                patched.modality,
                Path(str(entry["path"])).name,
            )
        return patched.data, byte_digest(patched.data)

    def _intrinsics_meta(
        self, backend: SourceBackedOutputBackend, entry: Mapping[str, Any], box: CropBox
    ) -> dict[str, Any]:
        attributes = entry.get("attributes")
        return {
            **(backend.modality_meta or {}),
            **(attributes if isinstance(attributes, Mapping) else {}),
            "width": box.output_width,
            "height": box.output_height,
        }

    def _camera(self, sample: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        intrinsics_name, extrinsics_name = self.projection or ("", "")
        intrinsics = to_numpy(single_calibration(sample[intrinsics_name], name=intrinsics_name))
        extrinsics = to_numpy(single_calibration(sample[extrinsics_name], name=extrinsics_name))
        if intrinsics.shape != (3, 3) or extrinsics.shape != (4, 4):
            raise ValueError(
                f"Sample {sample['full_id']} has intrinsics {intrinsics.shape} and extrinsics "
                f"{extrinsics.shape}; pruning needs 3x3 and 4x4"
            )
        return intrinsics, extrinsics

    def _write(
        self,
        name: str,
        full_id: str,
        entry: Mapping[str, Any],
        value: Any,
        box: CropBox,
        sample: Mapping[str, Any] | None = None,
    ) -> None:
        role = self.roles[name]
        backend = self.backends[name]
        attributes: dict[str, Any] = {"crop": box.to_dict()}

        if role.role == RASTER:
            size = role.raster_size(value)
            if size != (box.source_height, box.source_width):
                raise ValueError(
                    f"Modality '{name}' file {entry['path']!r} is {size[0]}x{size[1]}, but the "
                    f"reference '{self.config.reference}' is {box.source_height}x{box.source_width}"
                )
            cropped = role.slice(value, box.top, box.left, box.height, box.width)
            interpolation = "nearest"
            if box.resized:
                interpolation = self.config.interpolation_override(name) or role.default_interpolation()
                cropped = role.resize(cropped, box.output_height, box.output_width, interpolation)
            data, digest = self._encode(
                name, entry, cropped, attributes, tolerance=self._tolerance(role, interpolation)
            )
            backend.copy_source(full_id, entry, data, attributes=attributes)
            expectation = Expectation("value", digest)
        elif role.role == INTRINSICS:
            matrix = to_numpy(value)
            if matrix.shape != (3, 3):
                raise ValueError(
                    f"Modality '{name}' file {entry['path']!r} loaded a {matrix.shape} value, "
                    "not a 3x3 camera matrix"
                )
            data = self.sources.read(name, entry)
            recorded = recorded_image_size(data)
            if recorded is not None and recorded != (box.source_width, box.source_height):
                raise ValueError(
                    f"Modality '{name}' file {entry['path']!r} describes a {recorded[0]}x"
                    f"{recorded[1]} image, but the reference '{self.config.reference}' rasters "
                    f"are {box.source_width}x{box.source_height}: its matrix belongs to "
                    "another resolution"
                )
            cropped = crop_intrinsics(matrix, top=box.top, left=box.left)
            if box.resized:
                cropped = resize_intrinsics(
                    cropped,
                    source_size=(box.height, box.width),
                    target_size=(box.output_height, box.output_width),
                )
            patched = backend.updated_bytes(entry, data, cropped, self._intrinsics_meta(backend, entry, box))
            self._remember_patch(name, entry, data, patched, box)
            backend.copy_source(full_id, entry, patched, attributes=attributes)
            expectation = Expectation("intrinsics", cropped)
        elif self.prune and role.is_point_set and sample is not None:
            points = to_numpy(value)
            mask = visible_points(points, *self._camera(sample), box)
            kept = _select(value, mask)
            attributes["point_clouds"] = {
                "mode": "prune_to_crop",
                "source_points": int(len(mask)),
                "kept_points": int(mask.sum()),
            }
            data, digest = self._encode(name, entry, kept, attributes, tolerance=None)
            backend.copy_source(full_id, entry, data, attributes=attributes)
            expectation = Expectation("value", digest)
            self.points[name][0] += int(len(mask))
            self.points[name][1] += int(mask.sum())
        else:
            data, digest = self._copy_bytes(name, entry, value, box)
            backend.copy_source(full_id, entry, data, attributes=attributes)
            expectation = Expectation("bytes", digest)

        self.expectations[(name, full_id)] = expectation
        self.files_written[name] += 1
        self._boxes_by_modality[name].add(box)

    def crop_sample(self, sample: Mapping[str, Any]) -> bool:
        """Crop and write one sample's per-sample files; return False for a repeated sample."""
        full_id = str(sample["full_id"])
        if full_id in self._seen_samples:
            self.logger.warning("Sample %s is selected more than once; cropping it once", full_id)
            return False
        self._seen_samples.add(full_id)

        reference = self.config.reference
        height, width = self.roles[reference].raster_size(sample[reference])
        box = self.box_for(height, width)
        for name in self._write_order():
            if not self.roles[name].hierarchical:
                self._write(name, full_id, sample["meta"][name], sample[name], box, sample)
        self.hierarchical.add_sample(sample, box)
        return True

    def _write_hierarchical(self) -> None:
        for name in self._write_order():
            if not self.roles[name].hierarchical:
                continue
            files = self.hierarchical.files(name)
            if not files:
                raise ValueError(
                    f"No selected sample uses a file of hierarchical modality '{name}'"
                )
            for record in files:
                self._write(name, record.full_id, record.entry, record.value, record.box)

    def _finalize(self) -> None:
        spec = self.config.to_dict()
        for name, backend in self.backends.items():
            role = self.roles[name]
            boxes = sorted(
                self._boxes_by_modality[name],
                key=lambda box: (box.source_height, box.source_width),
            )
            record: dict[str, Any] = {
                "reference": self.config.reference,
                "spec": spec,
                "boxes": [box.to_dict() for box in boxes],
            }
            if self.prune and role.is_point_set and self.projection is not None:
                record["projection"] = {"intrinsics": self.projection[0], "extrinsics": self.projection[1]}
            updates: dict[str, Any] = {"crop": record}
            meta = backend.modality_meta or {}
            if role.role in (RASTER, INTRINSICS) and "dimensions" in meta:
                if len(boxes) == 1:
                    updates["dimensions"] = {
                        "height": boxes[0].output_height,
                        "width": boxes[0].output_width,
                    }
                else:
                    self.logger.warning(
                        "Dropping meta.dimensions of '%s': its files have %d crop sizes",
                        name,
                        len(boxes),
                    )
                    updates["dimensions"] = None
            backend.update_head_meta(updates)
            backend.finalize()

    def run(self, samples: Any) -> CropResult:
        total = len(samples)
        self.logger.info("crop: %d samples, modalities %s", total, ", ".join(
            f"{name} ({role.role}{', hierarchical' if role.hierarchical else ''})"
            for name, role in self.roles.items()
        ))
        cropped = 0
        try:
            for index in range(total):
                if self.crop_sample(samples[index]):
                    cropped += 1
                if (index + 1) % _PROGRESS_EVERY == 0:
                    self.logger.info("crop: %d/%d samples", index + 1, total)
            self._write_hierarchical()
            self._finalize()
        except Exception:
            self._close_writers()
            raise
        finally:
            self.sources.close()
        self.logger.info(
            "crop: wrote %s",
            ", ".join(f"{count} {name}" for name, count in self.files_written.items()),
        )
        for name, (source_points, kept_points) in self.points.items():
            self.logger.info(
                "crop: kept %d of %d %s points (%.1f %%) inside the crop",
                kept_points,
                source_points,
                name,
                100.0 * kept_points / max(source_points, 1),
            )
        return CropResult(
            samples=cropped,
            boxes=sorted(self._boxes.values(), key=lambda box: (box.source_height, box.source_width)),
            outputs=dict(self.outputs),
            roles=dict(self.roles),
            expectations=dict(self.expectations),
            files_written=dict(self.files_written),
            points={name: (values[0], values[1]) for name, values in self.points.items()},
        )


# -- command ------------------------------------------------------------------


def _resolve(path: str, config_dir: Path) -> str:
    candidate = Path(path)
    return str(candidate if candidate.is_absolute() else (config_dir / candidate).resolve())


def _source_roots(dataset: Any) -> list[Path]:
    paths = {**dataset.modality_paths(), **dataset.hierarchical_modality_paths()}
    return [Path(str(entry["path"])).resolve() for entry in paths.values()]


def resolve_output_locations(
    config: Mapping[str, Any], names: list[str], dataset: Any, config_dir: Path
) -> tuple[dict[str, OutputLocation], str | None]:
    """Return each modality's output location and the pipeline manifest path, if any.

    ``output_layout: "per_modality"`` (default) writes ``<output_path>/<name>``
    or the modality's ``pipeline.output_targets`` entry (slot = name; with
    targets, every modality needs one). ``"scoped"`` writes one root at
    ``output_path`` (a directory or ``.zip``) with one metadata scope per
    modality and its files below ``<name>/``.
    """
    layout = config.get("output_layout", "per_modality")
    if layout not in OUTPUT_LAYOUTS:
        raise ValueError(f"output_layout must be one of {', '.join(OUTPUT_LAYOUTS)}, got {layout!r}")
    pipeline = parse_pipeline_config(dict(config))
    targets = {target.slot: target for target in (pipeline.output_targets if pipeline else [])}
    unknown = sorted(set(targets) - set(names))
    if unknown:
        raise ValueError(f"pipeline.output_targets name unknown modalities: {', '.join(unknown)}")
    if targets:
        if layout == "scoped":
            raise ValueError("output_layout 'scoped' writes one root and cannot use pipeline.output_targets")
        missing = [name for name in names if name not in targets]
        if missing:
            raise ValueError(
                f"pipeline.output_targets has no slot for: {', '.join(missing)}; "
                "give every cropped modality a target"
            )

    output_path = config.get("output_path") or (pipeline.output_root if pipeline else None)
    locations: dict[str, OutputLocation] = {}
    if layout == "scoped":
        if not output_path:
            raise ValueError("output_layout 'scoped' needs output_path")
        root = Path(_resolve(str(output_path), config_dir))
        storage = "zip" if root.suffix.lower() == ".zip" else "directory"
        for name in names:
            locations[name] = OutputLocation(root, storage, metadata_scope=name, path_prefix=name)
    for name in names if layout == "per_modality" else []:
        target = targets.get(name)
        if target is not None:
            if target.storage == "file":
                raise ValueError(f"pipeline output target '{name}' uses unsupported storage='file'")
            source_type = _modality_key(dataset.get_modality_index(name))
            if source_type and target.dataset_type != source_type:
                # Callers such as euler-view send their canonical type, which may
                # be an alias of the head's modality key.
                get_logger().warning(
                    "pipeline output target '%s' declares datasetType '%s'; the source head says '%s'",
                    name,
                    target.dataset_type,
                    source_type,
                )
            locations[name] = OutputLocation(Path(target.path), target.storage, target)
        else:
            if not output_path:
                raise ValueError("output_path is required unless pipeline.output_targets covers every modality")
            locations[name] = OutputLocation(Path(_resolve(str(output_path), config_dir)) / name)

    roots = [location.root.resolve() for location in locations.values()]
    if layout == "per_modality" and len(set(roots)) != len(roots):
        raise ValueError("Two modalities would be written to the same output root")
    for root in set(roots):
        for source in _source_roots(dataset):
            if root == source or source in root.parents:
                raise ValueError(
                    f"Output {root} lies inside the source dataset {source}; choose an "
                    "output_path outside every input"
                )
    manifest_path = pipeline.outputs_manifest_path if pipeline and targets else None
    return locations, manifest_path


def _pack_zip(directory: Path, archive: Path) -> None:
    """Write *directory*'s files into the new *archive* (stored for compressed formats)."""
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w") as handle:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                compression = (
                    zipfile.ZIP_STORED
                    if path.suffix.lower() in COMPRESSED_EXTENSIONS
                    else zipfile.ZIP_DEFLATED
                )
                handle.write(path, path.relative_to(directory).as_posix(), compress_type=compression)


def run_crop_command(config: dict[str, Any], config_dir: Path) -> CropResult:
    """Run ``euler-preprocess crop`` for a parsed dataset config."""
    from euler_preprocess.cli import _select_configured_samples
    from euler_preprocess.common.dataset import build_dataset
    from euler_preprocess.crop.verify import verify_crop

    logger = get_logger()
    if config.get("preprocessing"):
        raise ValueError("The crop dataset config must not contain 'preprocessing'; it would crop twice")
    transform_config_path = config.get("transform_config_path")
    if not transform_config_path:
        raise ValueError("transform_config_path (the crop config) is required")
    crop_config_path = Path(_resolve(str(transform_config_path), config_dir))
    crop_config = parse_crop_config(json.loads(crop_config_path.read_text(encoding="utf-8")))
    logger.info("Crop config: %s", crop_config_path)

    configured = {**(config.get("modalities") or {}), **(config.get("hierarchical_modalities") or {})}
    if crop_config.reference not in configured:
        raise ValueError(
            f"The reference modality '{crop_config.reference}' is not in the dataset config "
            f"(configured: {', '.join(configured)})"
        )

    config = dict(config)
    for group in ("modalities", "hierarchical_modalities"):
        entries = {}
        for name, entry in (config.get(group) or {}).items():
            if isinstance(entry, str):
                entries[name] = _resolve(entry, config_dir)
            else:
                entries[name] = {**entry, "path": _resolve(str(entry["path"]), config_dir)}
        if entries:
            config[group] = entries

    dataset = build_dataset(config, {crop_config.reference})
    samples = _select_configured_samples(config, dataset, logger)
    names = list(dataset.modality_paths()) + list(dataset.hierarchical_modality_paths())
    outputs, manifest_path = resolve_output_locations(config, names, dataset, config_dir)

    # Scoped modalities share one archive, which one zip writer at a time owns:
    # write them into a staging directory and pack it once at the end.
    final_outputs = outputs
    staging: Path | None = None
    archive = next(iter(outputs.values())).root if outputs else None
    if archive is not None and any(
        location.metadata_scope and location.storage == "zip" for location in outputs.values()
    ):
        archive.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{archive.name}.", dir=archive.parent))
        outputs = {
            name: OutputLocation(staging, "directory", None, location.metadata_scope, location.path_prefix)
            for name, location in outputs.items()
        }
    for name, location in final_outputs.items():
        logger.info("Output [%s]: %s%s", name, location.root, f" (scope {location.metadata_scope})" if location.metadata_scope else "")

    try:
        result = CropRunner(dataset, crop_config, outputs).run(samples)
        if staging is not None and archive is not None:
            _pack_zip(staging, archive)
            result.outputs = dict(final_outputs)
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)

    if crop_config.verify != "none":
        limit = None if crop_config.verify == "all" else int(crop_config.verify)
        checked = verify_crop(result, limit=limit, reference=crop_config.reference)
        logger.info("crop: verified %d reloaded samples", checked)
    if manifest_path:
        write_pipeline_outputs_manifest(
            manifest_path,
            [location.target for location in final_outputs.values() if location.target is not None],
        )
    return result
