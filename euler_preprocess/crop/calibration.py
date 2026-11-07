"""Hierarchical modalities: files shared by many samples, such as calibration.

A hierarchical file (one ``calib_cam_stereo_left.json`` for every
SeeingThroughFog frame, per-scene intrinsics, ...) is written once, at its
original place in the hierarchy. All selected samples that use it must have
the same crop box; otherwise a single output file cannot describe them all.
Files that no selected sample uses are left out.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from euler_preprocess.crop.spec import CropBox

Level = tuple[str, ...]


def hierarchy_path(full_id: str) -> Level:
    """Return the hierarchy keys of a sample's ``full_id`` (``/key/.../file_id``)."""
    parts = [part for part in str(full_id).split("/") if part]
    return tuple(parts[:-1])


def full_id_of(level: Level, file_id: str) -> str:
    return "/" + "/".join(level + (file_id,))


def match_with_level(
    sample_level: Level, files_by_level: Mapping[Level, list[dict[str, Any]]]
) -> list[tuple[Level, dict[str, Any]]]:
    """Like ``euler_loading.indexing.match_hierarchical_files``, also returning each file's level.

    Levels are walked from the sample's own up to the root; for a file ID found
    at several levels the deepest one wins.
    """
    seen: set[str] = set()
    matched: list[tuple[Level, dict[str, Any]]] = []
    for depth in range(len(sample_level), -1, -1):
        level = sample_level[:depth]
        for entry in files_by_level.get(level, []):
            if entry["id"] not in seen:
                seen.add(entry["id"])
                matched.append((level, entry))
    return matched


@dataclass
class HierarchicalFile:
    """One source file of a hierarchical modality and the crop it receives."""

    modality: str
    level: Level
    entry: dict[str, Any]
    value: Any
    box: CropBox
    first_sample: str

    @property
    def full_id(self) -> str:
        return full_id_of(self.level, self.entry["id"])


class HierarchicalFiles:
    """Collect, per hierarchical modality, the files the selected samples use."""

    def __init__(self, dataset: Any, names: Iterable[str]) -> None:
        self.levels: dict[str, dict[Level, list[dict[str, Any]]]] = {
            name: dataset.get_hierarchical_files(name) for name in names
        }
        self._files: dict[tuple[str, Level, str], HierarchicalFile] = {}

    def add_sample(self, sample: Mapping[str, Any], box: CropBox) -> None:
        """Record the files *sample* uses; raise if a file already got another box."""
        sample_level = hierarchy_path(sample["full_id"])
        for name, files_by_level in self.levels.items():
            values = sample.get(name)
            for level, entry in match_with_level(sample_level, files_by_level):
                value = values.get(entry["id"]) if isinstance(values, Mapping) else values
                key = (name, level, entry["id"])
                existing = self._files.get(key)
                if existing is None:
                    self._files[key] = HierarchicalFile(
                        modality=name,
                        level=level,
                        entry=entry,
                        value=value,
                        box=box,
                        first_sample=str(sample["full_id"]),
                    )
                elif existing.box != box:
                    raise ValueError(
                        f"Hierarchical modality '{name}' file {entry['path']!r} is shared by "
                        f"samples with different crops: {existing.first_sample} "
                        f"({existing.box.describe()}) and {sample['full_id']} "
                        f"({box.describe()}). One output file cannot describe both; "
                        "crop each image size in its own run."
                    )

    def files(self, name: str) -> list[HierarchicalFile]:
        return [record for (modality, _, _), record in self._files.items() if modality == name]
