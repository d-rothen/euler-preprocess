"""CLI entry point for euler-preprocess.

Used both by ``python main.py`` and by the installed ``euler-preprocess`` console
script.  Supports subcommands: ``fog``, ``sky-depth``, ``radial``.
"""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

from euler_preprocess.common.dataset import build_dataset
from euler_preprocess.common.logging import get_logger, log_dataset_info
from euler_preprocess.common.output import prepare_output_backends


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve(path_str: str, config_dir: Path) -> Path:
    """Resolve a path relative to the config file's directory."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (config_dir / p).resolve()


def _run_transform(args: argparse.Namespace, transform_class: type) -> int:
    """Shared logic for all subcommands."""
    logger = get_logger()

    config_path = Path(args.config).resolve()
    config_dir = config_path.parent

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Resolve the transform-specific config path.
    # Support both ``transform_config_path`` and legacy ``fog_config_path``.
    transform_config_key = "transform_config_path"
    if transform_config_key not in config:
        transform_config_key = "fog_config_path"
    transform_config_path = _resolve(config[transform_config_key], config_dir)
    # Read the transform config to determine the device (for dataset logging)
    with open(transform_config_path, "r", encoding="utf-8") as f:
        transform_cfg = json.load(f)
    device = transform_cfg.get("device", "cpu").lower()
    use_gpu = device not in ("cpu",)

    logger.info("Config: %s", args.config)
    logger.info("Transform config: %s", transform_config_path)

    required_modalities = transform_class.REQUIRED_MODALITIES
    required_hierarchical = transform_class.REQUIRED_HIERARCHICAL_MODALITIES or None
    dataset = build_dataset(config, required_modalities, required_hierarchical)
    output_backends = prepare_output_backends(config, dataset, transform_class)
    primary_backend = next(iter(output_backends.values()))
    dataset_name = config.get("dataset", "dataset")

    raw_modalities = {
        **config.get("modalities", {}),
        **config.get("hierarchical_modalities", {}),
    }
    modality_info: dict[str, dict] = {}
    for name, entry in raw_modalities.items():
        if isinstance(entry, str):
            modality_info[name] = {"path": entry}
        else:
            modality_info[name] = entry
    log_dataset_info(logger, dataset_name, len(dataset), modality_info, use_gpu)
    for slot, backend in output_backends.items():
        logger.info("Output path [%s]: %s", slot, backend.root)

    transform_kwargs: dict = {
        "config_path": str(transform_config_path),
        "out_path": str(primary_backend.root),
    }
    init_params = inspect.signature(transform_class.__init__).parameters
    if "output_backends" in init_params:
        transform_kwargs["output_backends"] = output_backends
    else:
        transform_kwargs["output_backend"] = primary_backend
        if len(output_backends) > 1:
            extra = [s for s in output_backends if s != next(iter(output_backends))]
            logger.warning(
                "%s does not accept output_backends; ignoring auxiliary slots: %s",
                transform_class.__name__,
                extra,
            )
    if "strict" in init_params:
        transform_kwargs["strict"] = bool(getattr(args, "strict", False))
    elif getattr(args, "strict", False):
        logger.warning(
            "--strict was set but %s does not support strict mode; ignoring.",
            transform_class.__name__,
        )
    transform = transform_class(**transform_kwargs)

    saved_paths = transform.run(dataset)

    logger.info("Transform complete. Generated %d outputs.", len(saved_paths))
    return 0


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def _cmd_fog(args: argparse.Namespace) -> int:
    from euler_preprocess.fog.transform import FogTransform
    return _run_transform(args, FogTransform)


def _cmd_sky_depth(args: argparse.Namespace) -> int:
    from euler_preprocess.sky_depth.transform import SkyDepthTransform
    return _run_transform(args, SkyDepthTransform)


def _cmd_radial(args: argparse.Namespace) -> int:
    from euler_preprocess.radial.transform import RadialTransform
    return _run_transform(args, RadialTransform)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocessing transforms for RGB+depth datasets.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # Shared flags available to every subcommand.  Currently only honoured by
    # subcommands whose transforms accept a ``strict`` argument (see
    # ``_run_transform``); others log a warning and ignore it.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config", "-c", type=str, required=True,
        help="Path to the dataset configuration JSON file.",
    )
    common.add_argument(
        "--strict", action="store_true",
        help="Abort early if leading-sample sanity checks detect suspicious "
             "inputs (e.g. 0%% or ~100%% sky pixels, wrong mask shape). "
             "Currently honoured by: sky-depth.",
    )

    fog_parser = subparsers.add_parser(
        "fog", parents=[common], help="Apply synthetic fog to RGB images.",
    )
    fog_parser.set_defaults(func=_cmd_fog)

    sky_depth_parser = subparsers.add_parser(
        "sky-depth", parents=[common], help="Override sky-region depth values.",
    )
    sky_depth_parser.set_defaults(func=_cmd_sky_depth)

    radial_parser = subparsers.add_parser(
        "radial", parents=[common],
        help="Convert planar (z-buffer) depth to radial (Euclidean) depth.",
    )
    radial_parser.set_defaults(func=_cmd_radial)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return parser.parse_args(["--help"])
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    return args.func(args)
