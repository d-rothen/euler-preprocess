"""Every transform ``euler-preprocess`` can run: built-in and installed.

A :class:`~euler_preprocess.common.transform.Transform` subclass in any
installed package becomes runnable (``euler-preprocess run <name> -c ...``)
by declaring it in the suite's ``euler.transforms`` entry-point group (the
protocol of :mod:`euler_dataset_contract.extensions`)::

    [project.entry-points."euler.transforms"]
    thermal-noise = "my_lab.transforms:ThermalNoiseTransform"

Built-in transforms are listed in :data:`BUILTIN_TRANSFORMS` and loaded
through the same path, so every transform carries the distribution that
provides it and is checked by the same rules (:func:`validate_transform`).
Nothing is dropped silently: an installed transform that cannot be imported,
breaks a rule, takes the name of a built-in transform or command, or is
offered by two packages at once is not loaded, and the reason is kept as a
problem (:func:`transform_problems`) that ``euler-preprocess transforms`` and
the capability report show.
"""

from __future__ import annotations

import importlib.metadata
import inspect
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from euler_dataset_contract.extensions import (
    Extension,
    ExtensionError,
    ExtensionProblem,
    load_extensions,
    strict_mode,
)

from euler_preprocess.common.output import OutputSlotSpec, _resolve_primary_slot
from euler_preprocess.common.transform import Transform

logger = logging.getLogger(__name__)

TRANSFORMS_GROUP = "euler.transforms"
PACKAGE_NAME = "euler-preprocess"

#: The built-in transforms: CLI name -> ``module:Class``. Each also has its own
#: subcommand (``euler-preprocess fog``) besides ``euler-preprocess run fog``.
BUILTIN_TRANSFORMS: dict[str, str] = {
    "fog": "euler_preprocess.fog.transform:FogTransform",
    "lidar-motion": "euler_preprocess.lidar_motion.transform:LidarMotionTransform",
    "radial": "euler_preprocess.radial.transform:RadialTransform",
    "sky-depth": "euler_preprocess.sky_depth.transform:SkyDepthTransform",
}

#: Subcommands that are not transforms. A transform may not take their names.
COMMANDS: dict[str, str] = {
    "crop": "Crop RGB, depth and calibration coherently (K' = C K; lidar and "
    "extrinsics unchanged).",
    "crop-box": "Print the pixel box a crop config resolves to, and the matching "
    "spatial preprocessing block.",
    "spatial": "Export deterministic crop/resize with verified execution receipts.",
    "run": "Run any transform by name.",
    "transforms": "List the transforms this installation can run.",
}

_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

#: Constructor keyword arguments the CLI passes when a transform accepts them.
RUN_OPTIONS: dict[str, str] = {
    "output_backends": "receives every output slot's backend, so it can write "
    "auxiliary slots (otherwise only the primary slot's `output_backend`)",
    "depth_radial": "receives what the depth's dataset head declares in "
    "`radial_depth`",
    "strict": "honours `--strict`",
    "samples_per_variant": "accepts `file_ids` selection in the dataset config "
    "(then `samples` is the number of renders per file)",
}


def run_options(transform_class: type[Transform]) -> dict[str, bool]:
    """Which :data:`RUN_OPTIONS` the transform's constructor accepts."""
    parameters = inspect.signature(transform_class.__init__).parameters
    return {option: option in parameters for option in RUN_OPTIONS}


@dataclass(frozen=True)
class TransformEntry:
    """A runnable transform and the distribution that provides it."""

    name: str
    transform_class: type[Transform]
    extension: Extension
    builtin: bool

    def provider(self) -> dict[str, Any]:
        return {**self.extension.describe(), "builtin": self.builtin}


@dataclass(frozen=True)
class TransformCatalog:
    entries: dict[str, TransformEntry]
    problems: tuple[ExtensionProblem, ...]


def _string_set(value: Any, label: str) -> set[str]:
    if not isinstance(value, (set, frozenset)) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise TypeError(f"{label} must be a set of modality names")
    return set(value)


def validate_transform(extension: Extension, value: Any) -> type[Transform]:
    """Check that *value* is a transform the CLI can run; raise with the reason."""
    if not _NAME.fullmatch(extension.name):
        raise ValueError(
            f"transform name {extension.name!r} must be lower-case letters, digits, "
            "'-' and '_' (it is a CLI name)"
        )
    if not (inspect.isclass(value) and issubclass(value, Transform)):
        raise TypeError(f"{extension.target} is not a Transform subclass")
    if inspect.isabstract(value):
        missing = ", ".join(sorted(value.__abstractmethods__))
        raise TypeError(f"{extension.target} does not implement {missing}")

    label = value.__name__
    required = _string_set(value.REQUIRED_MODALITIES, f"{label}.REQUIRED_MODALITIES")
    optional = _string_set(value.OPTIONAL_MODALITIES, f"{label}.OPTIONAL_MODALITIES")
    hierarchical = _string_set(
        value.REQUIRED_HIERARCHICAL_MODALITIES,
        f"{label}.REQUIRED_HIERARCHICAL_MODALITIES",
    )
    if not required:
        raise ValueError(f"{label}.REQUIRED_MODALITIES must name at least one modality")
    source = value.SOURCE_MODALITY
    if source not in required:
        raise ValueError(
            f"{label}.SOURCE_MODALITY ({source!r}) must be one of its "
            f"REQUIRED_MODALITIES ({', '.join(sorted(required))}): outputs mirror "
            "its files"
        )
    _resolve_primary_slot(value)
    known = required | optional | hierarchical
    unknown = sorted(set(value.MODALITY_FUNCTIONS) - known)
    if unknown:
        raise ValueError(
            f"{label}.MODALITY_FUNCTIONS names modalities it does not take: "
            f"{', '.join(unknown)}"
        )
    slots = tuple(value.OUTPUT_SLOTS)
    if value.OUTPUT_SLOT_SPECS and not run_options(value)["output_backends"]:
        raise TypeError(
            f"{label} declares auxiliary OUTPUT_SLOT_SPECS, so its __init__ must "
            "accept output_backends to write them"
        )
    for slot, spec in value.OUTPUT_SLOT_SPECS.items():
        if slot not in slots:
            raise ValueError(f"{label}.OUTPUT_SLOT_SPECS[{slot!r}] is not in OUTPUT_SLOTS")
        if not isinstance(spec, OutputSlotSpec):
            raise TypeError(f"{label}.OUTPUT_SLOT_SPECS[{slot!r}] must be an OutputSlotSpec")
    hints = value.CONFIG_HINTS
    if not isinstance(hints, Mapping) or not all(
        isinstance(key, str) and isinstance(hint, Mapping) for key, hint in hints.items()
    ):
        raise TypeError(f"{label}.CONFIG_HINTS must map config keys to mappings")
    return value


def _package_version() -> str | None:
    try:
        return importlib.metadata.version(PACKAGE_NAME)
    except importlib.metadata.PackageNotFoundError:
        return None


def _builtin_entries() -> dict[str, TransformEntry]:
    version = _package_version()
    entries = {}
    for name, target in BUILTIN_TRANSFORMS.items():
        extension = Extension(
            TRANSFORMS_GROUP, name, target, distribution=PACKAGE_NAME, version=version
        )
        # A broken built-in is a bug in this package: raise, never hide it.
        transform_class = validate_transform(extension, extension.load())
        entries[name] = TransformEntry(name, transform_class, extension, True)
    return entries


def _shadowing(name: str, distributions: tuple[str | None, ...]) -> ExtensionProblem:
    kind = "transform" if name in BUILTIN_TRANSFORMS else "command"
    return ExtensionProblem(
        TRANSFORMS_GROUP,
        name,
        "shadows_builtin",
        f"{', '.join(str(d) for d in distributions)} provide(s) transform {name!r}, "
        f"which is a built-in {kind} of {PACKAGE_NAME}; the built-in is used. "
        "Rename the entry point.",
        distributions,
    )


def _load_catalog() -> TransformCatalog:
    entries = _builtin_entries()
    reserved = set(BUILTIN_TRANSFORMS) | set(COMMANDS)
    result = load_extensions(TRANSFORMS_GROUP, validate=validate_transform, strict=False)
    problems = list(result.problems)
    for problem in result.problems:
        if problem.code == "conflict" and problem.name in reserved:
            problems.append(_shadowing(problem.name, problem.distributions))
    for loaded in result.loaded:
        extension = loaded.extension
        if extension.name in reserved:
            problems.append(_shadowing(extension.name, (extension.distribution,)))
            continue
        entries[extension.name] = TransformEntry(extension.name, loaded.value, extension, False)
    problems.sort(key=lambda problem: (problem.name, problem.code))
    for problem in problems:
        logger.warning("transform extensions: %s", problem.message)
    if strict_mode() and problems:
        raise ExtensionError("; ".join(problem.message for problem in problems))
    return TransformCatalog(dict(sorted(entries.items())), tuple(problems))


_catalog: TransformCatalog | None = None


def transform_catalog() -> TransformCatalog:
    """The transforms of this environment, loaded once per process."""
    global _catalog
    if _catalog is None:
        _catalog = _load_catalog()
    return _catalog


def refresh_transforms() -> TransformCatalog:
    """Load the transforms again (after installing a package in this process)."""
    global _catalog
    _catalog = None
    return transform_catalog()


def transform_problems() -> list[ExtensionProblem]:
    """Why installed transforms were not loaded (conflicts, shadowing, errors)."""
    return list(transform_catalog().problems)


def get_transform(name: str) -> TransformEntry:
    """The transform called *name*; ``KeyError`` lists the known ones."""
    entries = transform_catalog().entries
    try:
        return entries[name]
    except KeyError:
        known = ", ".join(entries) or "none"
        hint = "".join(
            f"; {problem.message}" for problem in transform_problems() if problem.name == name
        )
        raise KeyError(f"unknown transform {name!r} (known: {known}){hint}") from None


__all__ = [
    "BUILTIN_TRANSFORMS",
    "COMMANDS",
    "PACKAGE_NAME",
    "RUN_OPTIONS",
    "TRANSFORMS_GROUP",
    "TransformCatalog",
    "TransformEntry",
    "get_transform",
    "refresh_transforms",
    "run_options",
    "transform_catalog",
    "transform_problems",
    "validate_transform",
]
