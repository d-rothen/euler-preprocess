"""Reload the crop's outputs and check them against what was written.

The output roots are reopened like any input (roles re-inferred from their
ds-crawler indexes) and every file is compared with the crop:

- rasters must reload bit-exactly as the cropped array (catches lossy writers);
- intrinsics must reload as ``K'`` to within 1e-3 px;
- copied files must be byte-identical to their source.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from euler_preprocess.crop.calibration import (
    HierarchicalFiles,
    full_id_of,
    hierarchy_path,
    match_with_level,
)
from euler_preprocess.crop.files import SourceFiles, byte_digest, fingerprint, to_numpy

INTRINSICS_TOLERANCE_PX = 1e-3


def _matches(expected: Any, value: Any, read_bytes) -> bool:
    if expected.kind == "value":
        return fingerprint(value) == expected.value
    if expected.kind == "intrinsics":
        matrix = to_numpy(value).astype(np.float64)
        return matrix.shape == (3, 3) and bool(
            np.all(np.abs(matrix - to_numpy(expected.value).astype(np.float64)) <= INTRINSICS_TOLERANCE_PX)
        )
    return byte_digest(read_bytes()) == expected.value


def verify_crop(result: Any, *, limit: int | None, reference: str) -> int:
    """Check the outputs of a :class:`~euler_preprocess.crop.run.CropResult`; return samples checked.

    *limit* checks only the first samples; ``None`` checks every sample and
    requires every written file to have been reloaded.
    """
    from euler_preprocess.common.dataset import build_dataset

    regular = [name for name, role in result.roles.items() if not role.hierarchical]
    hierarchical = [name for name, role in result.roles.items() if role.hierarchical]

    def entry(name: str) -> Any:
        location = result.outputs[name]
        if location.metadata_scope:
            return {"path": str(location.root), "metadata_scope": location.metadata_scope}
        return str(location.root)

    config = {"modalities": {name: entry(name) for name in result.roles}}
    output = build_dataset(config, set(regular) | {reference})

    reloaded_hierarchical = sorted(output.hierarchical_modality_paths())
    if reloaded_hierarchical != sorted(hierarchical):
        raise ValueError(
            f"The cropped outputs reload with hierarchical modalities {reloaded_hierarchical}, "
            f"but the inputs had {sorted(hierarchical)}"
        )
    if len(output) != result.samples:
        raise ValueError(
            f"The cropped outputs reload as {len(output)} samples, but {result.samples} were cropped"
        )

    sources = SourceFiles(output)
    levels = HierarchicalFiles(output, hierarchical).levels
    checked: set[tuple[str, str]] = set()

    def check(name: str, full_id: str, value: Any, entry: Mapping[str, Any]) -> None:
        key = (name, full_id)
        expected = result.expectations.get(key)
        if expected is None:
            raise ValueError(f"Output '{name}' holds an unexpected file {entry['path']!r} ({full_id})")
        if not _matches(expected, value, lambda: sources.read(name, entry)):
            raise ValueError(
                f"Verification failed: output '{name}' file {entry['path']!r} ({full_id}) "
                f"does not reload as the crop ({expected.kind} differs)"
            )
        checked.add(key)

    count = len(output) if limit is None else min(limit, len(output))
    try:
        for index in range(count):
            sample = output[index]
            full_id = str(sample["full_id"])
            for name in regular:
                check(name, full_id, sample[name], sample["meta"][name])
            for name in hierarchical:
                values = sample[name]
                for level, entry in match_with_level(hierarchy_path(full_id), levels[name]):
                    value = values.get(entry["id"]) if isinstance(values, Mapping) else values
                    check(name, full_id_of(level, entry["id"]), value, entry)
    finally:
        sources.close()

    if limit is None:
        unchecked = sorted(set(result.expectations) - checked)
        if unchecked:
            name, full_id = unchecked[0]
            raise ValueError(
                f"{len(unchecked)} written files were not reloaded, e.g. '{name}' {full_id}"
            )
    return count
