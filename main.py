"""Main entry point for fog generation using euler-loading.

Builds a MultiModalDataset from euler-loading, derives sky masks via a
transform, and feeds the samples into Foggify.

Usage:
    python main.py --config path/to/config.json

Example configuration:
    {
        "fog_config_path": "src/fog/example_config.json",
        "output_path": "/path/to/output",
        "modalities": {
            "rgb": "/path/to/vkitti_2.0.3_rgb",
            "depth": "/path/to/vkitti_2.0.3_depth",
            "classSegmentation": "/path/to/vkitti_2.0.3_classSegmentation"
        },
        "sky_color": [90, 200, 255]
    }
"""

import argparse
import json
import sys

import numpy as np
from PIL import Image

from euler_loading import Modality, MultiModalDataset
from src.fog.foggify import Foggify
from src.fog.sky_mask import sky_mask_transform


# ---------------------------------------------------------------------------
# Loader callables (one per modality)
# ---------------------------------------------------------------------------

def load_rgb(path: str) -> Image.Image:
    """Load an RGB image as a PIL Image."""
    return Image.open(path).convert("RGB")


def load_depth(path: str) -> np.ndarray:
    """Load a depth map as a float32 numpy array (values in meters)."""
    return np.asarray(Image.open(path), dtype=np.float32)


def load_class_segmentation(path: str) -> Image.Image:
    """Load a segmentation image as a PIL Image."""
    return Image.open(path).convert("RGB")


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


def main() -> int:
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    fog_config_path = config["fog_config_path"]
    output_path = config["output_path"]
    modality_paths = config["modalities"]
    sky_color = config.get("sky_color", [90, 200, 255])

    print(f"Loading configuration from: {args.config}")
    print(f"Fog config: {fog_config_path}")
    print(f"Output path: {output_path}")

    # Build euler-loading dataset
    dataset = MultiModalDataset(
        modalities={
            "rgb": Modality(modality_paths["rgb"], loader=load_rgb),
            "depth": Modality(modality_paths["depth"], loader=load_depth),
            "classSegmentation": Modality(
                modality_paths["classSegmentation"],
                loader=load_class_segmentation,
            ),
        },
        transforms=[sky_mask_transform(sky_color)],
    )

    print(f"Dataset size: {len(dataset)} samples")

    foggify = Foggify(
        config_path=fog_config_path,
        out_path=output_path,
    )

    saved_paths = foggify.generate_fog(dataset)

    print(f"\nFog generation complete. Generated {len(saved_paths)} images.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
