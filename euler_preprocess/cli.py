"""CLI entry point for euler-preprocess.

Used both by ``python main.py`` and by the installed ``euler-preprocess`` console
script.
"""

import argparse
import json
from pathlib import Path

from euler_preprocess.common.dataset import build_dataset
from euler_preprocess.common.logging import get_logger, log_dataset_info
from euler_preprocess.fog.foggify import Foggify


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_FOG_REQUIRED_MODALITIES = {"rgb", "depth", "sky_mask"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate foggy versions of datasets.",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to the configuration JSON file.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _resolve(path_str: str, config_dir: Path) -> Path:
    """Resolve a path relative to the config file's directory."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (config_dir / p).resolve()


def main() -> int:
    args = parse_args()
    logger = get_logger()

    config_path = Path(args.config).resolve()
    config_dir = config_path.parent

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    fog_config_path = _resolve(config["fog_config_path"], config_dir)
    output_path = config["output_path"]

    # Read the fog config to determine the device
    with open(fog_config_path, "r", encoding="utf-8") as f:
        fog_cfg = json.load(f)
    device = fog_cfg.get("device", "cpu").lower()
    use_gpu = device not in ("cpu",)

    logger.info("Config: %s", args.config)
    logger.info("Fog config: %s", fog_config_path)
    logger.info("Output path: %s", output_path)

    dataset = build_dataset(config, _FOG_REQUIRED_MODALITIES)
    dataset_name = config.get("dataset", "dataset")

    modality_paths = {
        **config.get("modalities", {}),
        **config.get("hierarchical_modalities", {}),
    }
    log_dataset_info(logger, dataset_name, len(dataset), modality_paths, use_gpu)

    foggify = Foggify(
        config_path=fog_config_path,
        out_path=output_path,
    )

    saved_paths = foggify.generate_fog(dataset)

    logger.info("Fog generation complete. Generated %d images.", len(saved_paths))
    return 0
