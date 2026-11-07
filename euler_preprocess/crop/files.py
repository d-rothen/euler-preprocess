"""Source-file access and value fingerprints for the crop."""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from ds_crawler.zip_utils import get_zip_root_prefix, is_zip_path

from euler_preprocess.common.normalize import _to_numpy


class SourceFiles:
    """Read a dataset's indexed files as bytes, from directories or zip archives."""

    def __init__(self, dataset: Any) -> None:
        paths = {**dataset.modality_paths(), **dataset.hierarchical_modality_paths()}
        self._roots = {name: str(entry["path"]) for name, entry in paths.items()}
        self._archives: dict[str, tuple[zipfile.ZipFile, str]] = {}

    def read(self, modality: str, entry: Mapping[str, Any]) -> bytes:
        """Return the bytes of the file *entry* of *modality* (``entry["path"]`` is root-relative)."""
        root = self._roots[modality]
        relative_path = str(entry["path"])
        if is_zip_path(root):
            if root not in self._archives:
                self._archives[root] = (zipfile.ZipFile(root, "r"), get_zip_root_prefix(Path(root)))
            archive, prefix = self._archives[root]
            return archive.read(prefix + relative_path)
        return (Path(root) / relative_path).read_bytes()

    def close(self) -> None:
        for archive, _prefix in self._archives.values():
            archive.close()
        self._archives.clear()


def to_numpy(value: Any) -> np.ndarray:
    return np.asarray(_to_numpy(value))


def fingerprint(value: Any) -> str:
    """Return a digest of an array's dtype, shape and contents."""
    array = np.ascontiguousarray(to_numpy(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(repr(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def byte_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def recorded_image_size(data: bytes) -> tuple[int, int] | None:
    """Return the ``(width, height)`` a JSON calibration file records, if any.

    ROS CameraInfo (SeeingThroughFog's ``calib_cam_stereo_left.json``) stores
    the image size its matrices describe as top-level ``width`` and ``height``;
    0 means unknown.
    """
    try:
        document = json.loads(data)
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(document, dict):
        return None
    width, height = document.get("width"), document.get("height")
    if (
        isinstance(width, int)
        and isinstance(height, int)
        and not isinstance(width, bool)
        and not isinstance(height, bool)
        and width > 0
        and height > 0
    ):
        return width, height
    return None
