"""CLI entry point for euler-preprocess.

Used both by ``python main.py`` and by the installed ``euler-preprocess`` console
script. ``run <name>`` runs any registered transform, built-in or provided by an
installed package through the ``euler.transforms`` entry-point group (see
:mod:`euler_preprocess.registry`); the built-in transforms ``fog``,
``sky-depth``, ``radial`` and ``lidar-motion`` also have their own subcommand.
``transforms`` lists them. ``crop``, ``crop-box`` and ``spatial`` are commands of
their own.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

from euler_preprocess.common.dataset import build_dataset, declared_radial_depth
from euler_preprocess.common.logging import get_logger, log_dataset_info
from euler_preprocess.common.output import prepare_output_backends
from euler_preprocess.registry import (
    BUILTIN_TRANSFORMS,
    COMMANDS,
    TransformEntry,
    get_transform,
    run_options,
    transform_catalog,
    transform_problems,
)


def _resolve(path_str: str, config_dir: Path) -> Path:
    """Resolve a path relative to the config file's directory."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (config_dir / p).resolve()


class _SelectedSamples(Sequence):
    """Lazy view over selected euler-loading dataset entries."""

    def __init__(self, dataset, indices: Iterable[int]) -> None:
        self.dataset = dataset
        self.indices = tuple(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self) -> Iterator[dict]:
        for index in self.indices:
            yield self.dataset[index]

    def __getitem__(self, index: int | slice):
        if isinstance(index, slice):
            return [self.dataset[i] for i in self.indices[index]]
        return self.dataset[self.indices[index]]


def _validate_sample_index(value: Any, *, key: str, dataset_size: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be a non-negative integer index")
    if value < 0:
        raise ValueError(f"{key} must be a non-negative integer index")
    if value >= dataset_size:
        raise IndexError(
            f"{key} {value} out of range for dataset of length {dataset_size}"
        )
    return value


def _positive_int(value: Any, *, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _non_negative_int(value: Any, *, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return value


def _resolve_sample_indices(selection: Any, *, dataset_size: int) -> tuple[int, ...]:
    if isinstance(selection, list):
        indices = tuple(
            _validate_sample_index(value, key="samples[]", dataset_size=dataset_size)
            for value in selection
        )
        if not indices:
            raise ValueError("samples must select at least one dataset entry")
        return indices

    if not isinstance(selection, dict):
        raise ValueError("samples must be an object or a list of integer indices")

    allowed = {"start", "stop", "step", "count"}
    unknown = sorted(set(selection) - allowed)
    if unknown:
        raise ValueError(f"samples contains unknown keys: {', '.join(unknown)}")

    start = _non_negative_int(selection.get("start", 0), key="samples.start")
    stop_value = selection.get("stop")
    if stop_value is None:
        stop = dataset_size
    else:
        stop = _non_negative_int(stop_value, key="samples.stop")
    step = _positive_int(selection.get("step", 1), key="samples.step")

    if start >= dataset_size:
        raise IndexError(
            f"samples.start {start} out of range for dataset of length {dataset_size}"
        )

    indices = tuple(range(start, min(stop, dataset_size), step))
    if "count" in selection:
        count = _positive_int(selection["count"], key="samples.count")
        indices = indices[:count]

    if not indices:
        raise ValueError("samples must select at least one dataset entry")
    return indices


def _select_configured_samples(config: dict, dataset, logger):
    """Apply optional top-level sample selection from the dataset config."""
    has_sample = "sample" in config
    has_samples = "samples" in config
    if has_sample and has_samples:
        raise ValueError("Use either sample or samples, not both")
    if not has_sample and not has_samples:
        return dataset

    dataset_size = len(dataset)
    if has_sample:
        sample_index = _validate_sample_index(
            config["sample"],
            key="sample",
            dataset_size=dataset_size,
        )
        sample = dataset[sample_index]
        logger.info(
            "Sample selection: using sample=%d of %d (id=%s, full_id=%s)",
            sample_index,
            dataset_size,
            sample.get("id"),
            sample.get("full_id"),
        )
        return [sample]

    indices = _resolve_sample_indices(config["samples"], dataset_size=dataset_size)
    logger.info(
        "Sample selection: using %d/%d samples (first_index=%d, last_index=%d)",
        len(indices),
        dataset_size,
        indices[0],
        indices[-1],
    )
    return _SelectedSamples(dataset, indices)


def _select_file_ids(
    file_ids: Any, dataset, logger, source_modality: str = "rgb",
) -> _SelectedSamples:
    """Select samples by file ID of *source_modality* (short or qualified)."""
    if not isinstance(file_ids, list) or not file_ids or any(
        not isinstance(file_id, str) or not file_id.strip() for file_id in file_ids
    ):
        raise ValueError("file_ids must be a non-empty list of file ID strings")
    normalized = [
        "/" + file_id.lstrip("/") if "/" in file_id else file_id
        for file_id in file_ids
    ]
    matches: dict[str, list[int]] = {file_id: [] for file_id in normalized}
    for index in range(len(dataset)):
        full_id = dataset.sample_full_id(index, source_modality)
        for key in (full_id, full_id.rsplit("/", 1)[-1]):
            if key in matches:
                matches[key].append(index)

    indices: list[int] = []
    for file_id, key in zip(file_ids, normalized):
        candidates = matches[key]
        if not candidates:
            raise ValueError(f"File ID {file_id!r} is absent from the joined dataset")
        if len(candidates) > 1:
            raise ValueError(f"File ID {file_id!r} is ambiguous; use its qualified ID")
        indices.append(candidates[0])
    if len(set(indices)) != len(indices):
        raise ValueError("file_ids contains duplicate source files")
    logger.info("File ID selection: using %d/%d samples", len(indices), len(dataset))
    return _SelectedSamples(dataset, indices)


def _transform_config_path(
    config: dict, config_dir: Path, transform_config: str | None,
) -> Path:
    if transform_config is not None:
        return Path(transform_config).resolve()
    if "transform_config_path" not in config:
        raise ValueError(
            "The dataset config has no transform_config_path; set it (relative to "
            "the dataset config) or pass --transform-config."
        )
    return _resolve(config["transform_config_path"], config_dir)


def run_transform(
    transform: str | TransformEntry,
    config: str | Path,
    *,
    transform_config: str | Path | None = None,
    strict: bool = False,
) -> list[Path]:
    """Run a registered transform on the dataset a dataset config describes.

    *transform* is a name from ``euler-preprocess transforms`` (built-in or
    from an installed package). *transform_config* replaces the dataset
    config's ``transform_config_path``. The transform receives the optional
    constructor keywords it accepts (:data:`~euler_preprocess.registry.RUN_OPTIONS`);
    asking for one it does not accept (``strict``, ``file_ids``) is an error.
    Returns the paths the transform wrote.
    """
    entry = get_transform(transform) if isinstance(transform, str) else transform
    transform_class = entry.transform_class
    options = run_options(transform_class)
    logger = get_logger()

    config_path = Path(config).resolve()
    config_dir = config_path.parent

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    transform_config_path = _transform_config_path(
        config, config_dir, None if transform_config is None else str(transform_config),
    )
    # Read the transform config to determine the device (for dataset logging)
    with open(transform_config_path, "r", encoding="utf-8") as f:
        transform_cfg = json.load(f)
    device = str(transform_cfg.get("device", "cpu")).lower()
    use_gpu = device not in ("cpu",)

    if strict and not options["strict"]:
        raise ValueError(
            f"--strict was set, but transform {entry.name!r} has no strict mode."
        )
    if "file_ids" in config and not options["samples_per_variant"]:
        raise ValueError(
            f"file_ids selection is not supported by transform {entry.name!r}; "
            "select samples with sample or samples."
        )

    logger.info("Transform: %s (%s)", entry.name, entry.extension.distribution)
    logger.info("Config: %s", config_path)
    logger.info("Transform config: %s", transform_config_path)

    required_modalities = transform_class.REQUIRED_MODALITIES
    required_hierarchical = transform_class.REQUIRED_HIERARCHICAL_MODALITIES or None
    # euler-loading logs which loader (and sky definition) each modality uses.
    get_logger("euler_loading")
    dataset = build_dataset(
        config,
        required_modalities,
        required_hierarchical,
        transform_class.MODALITY_FUNCTIONS,
        optional_modalities=transform_class.OPTIONAL_MODALITIES,
    )
    repeats = None
    if "file_ids" in config:
        if "sample" in config:
            raise ValueError("Use either file_ids or sample, not both")
        repeats = _positive_int(config.get("samples", 1), key="samples")
        samples = _select_file_ids(
            config["file_ids"], dataset, logger, transform_class.SOURCE_MODALITY,
        )
    else:
        samples = _select_configured_samples(config, dataset, logger)
    output_backends = prepare_output_backends(config, dataset, transform_class)
    primary_backend = next(iter(output_backends.values()))
    dataset_name = config.get("dataset", "dataset")

    raw_modalities = {
        **config.get("modalities", {}),
        **config.get("hierarchical_modalities", {}),
    }
    modality_info: dict[str, dict] = {}
    for name, entry_config in raw_modalities.items():
        if isinstance(entry_config, str):
            modality_info[name] = {"path": entry_config}
        else:
            modality_info[name] = entry_config
    log_dataset_info(logger, dataset_name, len(samples), modality_info, use_gpu)
    for slot, backend in output_backends.items():
        logger.info("Output path [%s]: %s", slot, backend.root)

    transform_kwargs: dict = {
        "config_path": str(transform_config_path),
        "out_path": str(primary_backend.root),
    }
    if repeats is not None:
        transform_kwargs["samples_per_variant"] = repeats
    if options["output_backends"]:
        transform_kwargs["output_backends"] = output_backends
    else:
        # validate_transform guarantees: no auxiliary slots without output_backends.
        transform_kwargs["output_backend"] = primary_backend
    if options["depth_radial"] and "depth" in dataset.modality_paths():
        depth_radial, source = declared_radial_depth(dataset, "depth")
        logger.info(
            "Depth: radial_depth=%s (from the %s)",
            "undeclared" if depth_radial is None else str(depth_radial).lower(),
            source,
        )
        transform_kwargs["depth_radial"] = depth_radial
    if options["strict"]:
        transform_kwargs["strict"] = bool(strict)
    instance = transform_class(**transform_kwargs)

    saved_paths = instance.run(samples)

    logger.info("Transform complete. Generated %d outputs.", len(saved_paths))
    return saved_paths


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        entry = get_transform(args.transform)
    except KeyError as exc:
        print(f"euler-preprocess: error: {exc.args[0]}", file=sys.stderr)
        return 2
    run_transform(
        entry,
        args.config,
        transform_config=args.transform_config,
        strict=args.strict,
    )
    return 0


def _describe_entry(entry: TransformEntry) -> list[str]:
    cls = entry.transform_class
    origin = "built-in" if entry.builtin else (
        f"{entry.extension.distribution or 'unknown package'} "
        f"{entry.extension.version or ''}".strip()
    )
    summary = (cls.__doc__ or "").strip().splitlines()
    lines = [f"{entry.name:<14} {summary[0] if summary else cls.__name__}  [{origin}]"]
    needs = ", ".join(sorted(cls.REQUIRED_MODALITIES | cls.REQUIRED_HIERARCHICAL_MODALITIES))
    lines.append(f"{'':<14} requires {needs}")
    if cls.OPTIONAL_MODALITIES:
        lines.append(f"{'':<14} optional {', '.join(sorted(cls.OPTIONAL_MODALITIES))}")
    slots = list(cls.OUTPUT_SLOTS) or [cls.OUTPUT_SLOT or cls.SOURCE_MODALITY]
    lines.append(f"{'':<14} writes   {', '.join(slots)}")
    return lines


def _cmd_transforms(args: argparse.Namespace) -> int:
    if args.json:
        from euler_preprocess.capabilities import transform_capability_report

        print(json.dumps(transform_capability_report(), indent=2))
        return 0
    for entry in transform_catalog().entries.values():
        print("\n".join(_describe_entry(entry)))
    problems = transform_problems()
    if problems:
        print(f"\n{len(problems)} installed transform(s) not loaded:")
        for problem in problems:
            print(f"  [{problem.code}] {problem.message}")
    return 0


def _cmd_crop(args: argparse.Namespace) -> int:
    from euler_preprocess.crop.run import run_crop_command

    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    get_logger().info("Config: %s", config_path)
    run_crop_command(config, config_path.parent)
    return 0


def _cmd_crop_box(args: argparse.Namespace) -> int:
    from euler_preprocess.crop.spec import parse_crop_config, spatial_preprocessing

    config = parse_crop_config(json.loads(Path(args.config).read_text(encoding="utf-8")))
    height, width = args.size
    box = config.geometry(height, width)
    print(json.dumps(
        {"box": box.to_dict(), "preprocessing": spatial_preprocessing(box, reference=config.reference)},
        indent=2,
    ))
    return 0


def _cmd_spatial(args: argparse.Namespace) -> int:
    from euler_preprocess.spatial import export_spatial

    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for group in ("modalities", "hierarchical_modalities"):
        for key, entry in config.get(group, {}).items():
            if isinstance(entry, str):
                config[group][key] = str(_resolve(entry, config_path.parent))
            else:
                entry["path"] = str(_resolve(entry["path"], config_path.parent))
    for entry in config["outputs"].values():
        entry["path"] = str(_resolve(entry["path"], config_path.parent))
    export_spatial(config)
    return 0


_BUILTIN_HELP = {
    "fog": "Apply synthetic fog to RGB images.",
    "sky-depth": "Override sky-region depth values.",
    "radial": "Convert planar (z-buffer) depth to radial (Euclidean) depth.",
    "lidar-motion": "Compensate scanning-lidar points for ego motion during the sweep.",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="euler-preprocess",
        description="Preprocessing transforms for RGB+depth datasets.",
    )
    subparsers = parser.add_subparsers(dest="command")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config", "-c", type=str, required=True,
        help="Path to the dataset configuration JSON file.",
    )
    common.add_argument(
        "--strict", action="store_true",
        help="Abort early if leading-sample sanity checks detect suspicious "
             "inputs (e.g. 0%% or ~100%% sky pixels, wrong mask shape). Only "
             "for transforms with a strict mode (sky-depth); others refuse it.",
    )
    transform_options = argparse.ArgumentParser(add_help=False)
    transform_options.add_argument(
        "--transform-config", "-t", type=str, default=None,
        help="Path to the transform config JSON file; replaces the dataset "
             "config's transform_config_path.",
    )

    run_parser = subparsers.add_parser(
        "run", parents=[common, transform_options], help=COMMANDS["run"],
        description="Run a transform by name: a built-in one or one an installed "
                    "package provides (euler.transforms entry points). "
                    "`euler-preprocess transforms` lists them.",
    )
    run_parser.add_argument("transform", metavar="NAME", help="Transform name.")
    run_parser.set_defaults(func=_cmd_run)

    for name in BUILTIN_TRANSFORMS:
        builtin_parser = subparsers.add_parser(
            name, parents=[common, transform_options], help=_BUILTIN_HELP[name],
        )
        builtin_parser.set_defaults(func=_cmd_run, transform=name)

    transforms_parser = subparsers.add_parser("transforms", help=COMMANDS["transforms"])
    transforms_parser.add_argument(
        "--json", action="store_true",
        help="Print the transform_capability_report (what euler-view reads).",
    )
    transforms_parser.set_defaults(func=_cmd_transforms)

    crop_parser = subparsers.add_parser("crop", parents=[common], help=COMMANDS["crop"])
    crop_parser.set_defaults(func=_cmd_crop)

    crop_box_parser = subparsers.add_parser("crop-box", help=COMMANDS["crop-box"])
    crop_box_parser.add_argument(
        "--config", "-c", type=str, required=True, help="Path to the crop config JSON file.",
    )
    crop_box_parser.add_argument(
        "--size", type=int, nargs=2, required=True, metavar=("HEIGHT", "WIDTH"),
        help="Size of the reference raster in pixels.",
    )
    crop_box_parser.set_defaults(func=_cmd_crop_box)

    spatial_parser = subparsers.add_parser(
        "spatial", parents=[common], help=COMMANDS["spatial"],
    )
    spatial_parser.set_defaults(func=_cmd_spatial)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    return args.func(args)
