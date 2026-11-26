"""What this installation can run: ``transform_capability_report`` 1.0.

Registered as ``euler-preprocess`` in the suite's ``euler.capabilities``
entry-point group, so ``euler-suite report`` (and euler-view, per cluster
environment) sees every transform it can run, built-in and from installed
packages, with the distribution providing each, the modalities each needs,
the output slots it writes, its transform-config keys, and the problems that
kept installed transforms out. ``euler-preprocess transforms --json`` prints
the same report.
"""

from __future__ import annotations

import json
from typing import Any

from euler_preprocess.common.output import _resolve_primary_slot
from euler_preprocess.registry import (
    COMMANDS,
    PACKAGE_NAME,
    RUN_OPTIONS,
    TRANSFORMS_GROUP,
    TransformEntry,
    _package_version,
    run_options,
    transform_catalog,
)

REPORT_KIND = "transform_capability_report"
REPORT_VERSION = "1.0"


def _json_safe(value: Any) -> Any:
    """*value* as JSON, or its ``str`` where it is not JSON-serializable."""
    try:
        return json.loads(json.dumps(value))
    except (TypeError, ValueError):
        return str(value)


def _summary(transform_class: type) -> str:
    lines = (transform_class.__doc__ or "").strip().splitlines()
    return lines[0].strip() if lines else ""


def _outputs(transform_class: type) -> list[dict[str, Any]]:
    primary = _resolve_primary_slot(transform_class)
    slots = list(transform_class.OUTPUT_SLOTS) or [primary]
    outputs = []
    for slot in slots:
        spec = transform_class.OUTPUT_SLOT_SPECS.get(slot)
        entry: dict[str, Any] = {"slot": slot, "primary": slot == primary}
        if slot == primary:
            entry["mirrors"] = transform_class.SOURCE_MODALITY
            entry["meta_overrides"] = _json_safe(
                dict(transform_class.OUTPUT_INDEX_META_OVERRIDES)
            )
        elif spec is not None:
            # Auxiliary slots are written only when pipeline.output_targets asks.
            entry.update(
                mirrors=spec.source_modality,
                modality_key=spec.index_overlay.get("type"),
                file_extension=spec.output_extension,
                opt_in=True,
            )
            loading = spec.index_overlay.get("euler_loading")
            if loading:
                entry["euler_loading"] = _json_safe(loading)
        outputs.append(entry)
    return outputs


def _transform(entry: TransformEntry) -> dict[str, Any]:
    cls = entry.transform_class
    return {
        "name": entry.name,
        "class": f"{cls.__module__}:{cls.__qualname__}",
        "summary": _summary(cls),
        "builtin": entry.builtin,
        "provider": entry.extension.describe(),
        "command": f"euler-preprocess run {entry.name} -c <dataset-config>",
        "modalities": {
            "required": sorted(cls.REQUIRED_MODALITIES),
            "optional": sorted(cls.OPTIONAL_MODALITIES),
            "required_hierarchical": sorted(cls.REQUIRED_HIERARCHICAL_MODALITIES),
            "loader_functions": dict(sorted(cls.MODALITY_FUNCTIONS.items())),
            "source": cls.SOURCE_MODALITY,
        },
        "outputs": _outputs(cls),
        "config": {key: _json_safe(dict(hint)) for key, hint in cls.CONFIG_HINTS.items()},
        "run_options": run_options(cls),
    }


def transform_capability_report() -> dict[str, Any]:
    """Every transform this environment can run, and why others cannot run."""
    catalog = transform_catalog()
    return {
        "contract": {"kind": REPORT_KIND, "version": REPORT_VERSION},
        "package": {"name": PACKAGE_NAME, "version": _package_version()},
        "extension_group": TRANSFORMS_GROUP,
        "transforms": [_transform(entry) for entry in catalog.entries.values()],
        "run_options": dict(RUN_OPTIONS),
        "commands": [
            {"name": name, "summary": summary} for name, summary in COMMANDS.items()
        ],
        "problems": [problem.describe() for problem in catalog.problems],
    }


__all__ = ["REPORT_KIND", "REPORT_VERSION", "transform_capability_report"]
