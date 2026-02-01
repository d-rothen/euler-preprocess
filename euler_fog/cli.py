"""CLI entry point for euler-fog.

Used both by ``python main.py`` and by the installed ``euler-fog`` console
script.
"""

import argparse
import importlib
import json
import sys
from pathlib import Path

from euler_fog.fog.foggify import Foggify
from euler_fog.fog.foggify_logging import get_logger, log_dataset_info


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
# Fallback dataset builder (vkitti2)
# ---------------------------------------------------------------------------

def _build_vkitti2_dataset(config: dict, use_gpu: bool):
    """Build a MultiModalDataset using the built-in vkitti2 loaders."""
    from euler_loading import Modality, MultiModalDataset
    from euler_fog.fog.sky_mask import sky_mask_transform

    if use_gpu:
        from euler_loading.loaders.gpu import vkitti2 as loaders
    else:
        from euler_loading.loaders.cpu import vkitti2 as loaders

    modality_paths = config["modalities"]
    sky_color = config.get("sky_color", [90, 200, 255])

    return MultiModalDataset(
        modalities={
            "rgb": Modality(modality_paths["rgb"], loader=loaders.rgb),
            "depth": Modality(modality_paths["depth"], loader=loaders.depth),
            "classSegmentation": Modality(
                modality_paths["classSegmentation"],
                loader=loaders.class_segmentation,
            ),
        },
        transforms=[sky_mask_transform(sky_color)],
    )


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
    dataset_name = config.get("dataset")
    modality_paths = config["modalities"]

    # Read the fog config to determine the device
    with open(fog_config_path, "r", encoding="utf-8") as f:
        fog_cfg = json.load(f)
    device = fog_cfg.get("device", "cpu").lower()
    use_gpu = device not in ("cpu",)

    logger.info("Config: %s", args.config)
    logger.info("Fog config: %s", fog_config_path)
    logger.info("Output path: %s", output_path)

    # Build dataset via the named module or fall back to vkitti2
    if dataset_name:
        module = importlib.import_module(f"euler_fog.common.{dataset_name}")
        dataset = module.build_dataset(config, use_gpu)
    else:
        dataset_name = "vkitti2"
        dataset = _build_vkitti2_dataset(config, use_gpu)

    log_dataset_info(logger, dataset_name, len(dataset), modality_paths, use_gpu)

    foggify = Foggify(
        config_path=fog_config_path,
        out_path=output_path,
    )

    saved_paths = foggify.generate_fog(dataset)

    logger.info("Fog generation complete. Generated %d images.", len(saved_paths))
    return 0
