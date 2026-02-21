from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_image(path: Path, rgb: np.ndarray) -> None:
    rgb = np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    img = Image.fromarray(rgb, mode="RGB")
    img.save(path)


def save_depth_npy(path: Path, depth: np.ndarray) -> None:
    """Save a float32 depth map as a ``.npy`` file."""
    np.save(path, depth.astype(np.float32))
