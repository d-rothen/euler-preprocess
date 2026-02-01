"""Main entry point for fog generation using euler-loading.

Builds a MultiModalDataset, derives sky masks via a dataset-specific
transform, and feeds the samples into Foggify.

Usage:
    python main.py --config path/to/config.json

The configuration JSON must contain at least ``fog_config_path``,
``output_path``, and ``modalities``.  An optional ``dataset`` key selects
which dataset builder to use (maps to a module under ``src/common/``).
Configs without ``dataset`` fall back to the built-in vkitti2 loaders.
"""

import argparse
import importlib
import json
import sys

from src.fog.foggify import Foggify
from src.fog.foggify_logging import get_logger, log_dataset_info


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate foggy versions of datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
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
    from src.fog.sky_mask import sky_mask_transform

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

def main() -> int:
    args = parse_args()
    logger = get_logger()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    fog_config_path = config["fog_config_path"]
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
        module = importlib.import_module(f"src.common.{dataset_name}")
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


if __name__ == "__main__":
    sys.exit(main())
