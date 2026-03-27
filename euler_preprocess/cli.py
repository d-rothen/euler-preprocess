"""CLI entry point for euler-preprocess.

Used both by ``python main.py`` and by the installed ``euler-preprocess`` console
script.  Supports subcommands: ``fog``, ``sky-depth``, ``radial``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from euler_preprocess.common.dataset import build_dataset
from euler_preprocess.common.logging import get_logger, log_dataset_info
from euler_preprocess.common.output import prepare_output_backend


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
    output_backend = prepare_output_backend(config, dataset, transform_class)
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
    logger.info("Output path: %s", output_backend.root)

    transform = transform_class(
        config_path=str(transform_config_path),
        out_path=str(output_backend.root),
        output_backend=output_backend,
    )

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

    # --- fog ---
    fog_parser = subparsers.add_parser(
        "fog", help="Apply synthetic fog to RGB images.",
    )
    fog_parser.add_argument(
        "--config", "-c", type=str, required=True,
        help="Path to the dataset configuration JSON file.",
    )
    fog_parser.set_defaults(func=_cmd_fog)

    # --- sky-depth ---
    sky_depth_parser = subparsers.add_parser(
        "sky-depth", help="Override sky-region depth values.",
    )
    sky_depth_parser.add_argument(
        "--config", "-c", type=str, required=True,
        help="Path to the dataset configuration JSON file.",
    )
    sky_depth_parser.set_defaults(func=_cmd_sky_depth)

    # --- radial ---
    radial_parser = subparsers.add_parser(
        "radial", help="Convert planar (z-buffer) depth to radial (Euclidean) depth.",
    )
    radial_parser.add_argument(
        "--config", "-c", type=str, required=True,
        help="Path to the dataset configuration JSON file.",
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
