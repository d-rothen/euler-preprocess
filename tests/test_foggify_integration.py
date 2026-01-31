"""Integration test for the fog generation pipeline.

Fill in the three paths below and implement the three loader stubs,
then run with:

    pytest tests/test_foggify_integration.py -m integration
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.fog.foggify import Foggify
from src.fog.sky_mask import sky_mask_transform
from typing import Any, Callable

# ── Loaders (implement before running) ─────────────────────────────────────
# Each loader receives the absolute path to a single file and must return
# the loaded data.  See the docstrings for the expected return types.


def load_rgb(path: str) -> Image.Image:
    """Load an RGB image as a PIL Image."""
    return Image.open(path).convert("RGB")


def load_depth(path: str) -> np.ndarray:
    """Load a depth map.

    Returns:
        np.ndarray, shape (H, W), dtype float32, values in **meters**.
    """
    return np.array(Image.open(path), dtype=np.float32)


def load_class_segmentation(path: str) -> Image.Image:
    """Load a segmentation image as a PIL Image."""
    return Image.open(path).convert("RGB")




REAL_DATASETS: dict[str, dict[str, Any]] = {
    "VKITTI2": {
        "modalities": {
            "rgb":  { "path": "/Volumes/Volume/Datasets/vkitti2/vkitti_2.0.3_rgb",   "loader": load_rgb },
            "depth": { "path": "/Volumes/Volume/Datasets/vkitti2/vkitti_2.0.3_depth", "loader": load_depth },
            "classSegmentation": { "path": "/Volumes/Volume/Datasets/vkitti2/vkitti_2.0.3_classSegmentation", "loader": load_class_segmentation },
        },
        "read_intrinsics": None,
        "read_extrinsics": None,
        "transforms": None,
    },
}

# ── Modality root paths (fill in before running) ───────────────────────────

RGB_PATH = REAL_DATASETS["VKITTI2"]["modalities"]["rgb"]["path"]
DEPTH_PATH = REAL_DATASETS["VKITTI2"]["modalities"]["depth"]["path"]
SEGMENTATION_PATH = REAL_DATASETS["VKITTI2"]["modalities"]["classSegmentation"]["path"]

# ── Constants ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_CONFIG_PATH = PROJECT_ROOT / "configs" / "example_dataset_config.json"
FOG_CONFIG_PATH = PROJECT_ROOT / "src" / "fog" / "example_config.json"
OUTPUT_DIR = PROJECT_ROOT / ".tests"


# ── Helpers ────────────────────────────────────────────────────────────────

def _collect_file_ids(
    rgb_root: Path, depth_root: Path, seg_root: Path
) -> list[tuple[Path, Path, Path]]:
    """Walk rgb_root and find matching files in depth_root / seg_root.

    Returns a list of (rgb_file, depth_file, seg_file) tuples for every
    file that exists in all three roots (matched by relative path stem).
    Adjust the path-mapping logic in this helper to match your dataset
    layout.
    """
    triples = []
    for rgb_file in sorted(rgb_root.rglob("*")):
        if not rgb_file.is_file():
            continue
        rel = rgb_file.relative_to(rgb_root)

        # Naive same-relative-path lookup – override if your dataset uses
        # different naming conventions between modalities.
        depth_file = depth_root / rel
        seg_file = seg_root / rel

        if depth_file.exists() and seg_file.exists():
            triples.append((rgb_file, depth_file, seg_file))
    return triples


def _build_samples(
    triples: list[tuple[Path, Path, Path]],
    sky_color: list[int],
) -> list[dict]:
    """Load files via the user-provided loaders and apply sky_mask_transform."""
    add_sky_mask = sky_mask_transform(sky_color)
    samples = []
    for rgb_file, depth_file, seg_file in triples:
        sample = {
            "rgb": load_rgb(str(rgb_file)),
            "depth": load_depth(str(depth_file)),
            "classSegmentation": load_class_segmentation(str(seg_file)),
            "id": rgb_file.stem,
        }
        sample = add_sky_mask(sample)
        samples.append(sample)
    return samples


def _make_cpu_fog_config(src: Path, dst: Path) -> Path:
    """Copy the fog config, forcing device to 'cpu' for test portability."""
    with open(src, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["device"] = "cpu"
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    return dst


# ── Test ───────────────────────────────────────────────────────────────────

@pytest.mark.integration
def test_foggify_with_real_data(tmp_path):
    """Run Foggify on real data and verify the outputs land in .tests/."""
    # ── read dataset config for sky_color ──
    with open(DATASET_CONFIG_PATH, "r", encoding="utf-8") as f:
        ds_cfg = json.load(f)
    sky_color = ds_cfg.get("sky_color", [90, 200, 255])

    # ── collect matching files across the three modalities ──
    triples = _collect_file_ids(
        Path(RGB_PATH), Path(DEPTH_PATH), Path(SEGMENTATION_PATH)
    )
    assert len(triples) > 0, (
        f"No matching files found across:\n"
        f"  rgb:  {RGB_PATH}\n"
        f"  depth: {DEPTH_PATH}\n"
        f"  seg:  {SEGMENTATION_PATH}"
    )

    # ── build sample dicts ──
    samples = _build_samples(triples, sky_color)

    # ── prepare output dir ──
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    # ── write a CPU-only copy of the fog config ──
    cpu_fog_config = _make_cpu_fog_config(
        FOG_CONFIG_PATH, tmp_path / "fog_config_cpu.json"
    )

    # ── run fog generation ──
    foggify = Foggify(
        config_path=str(cpu_fog_config),
        out_path=str(OUTPUT_DIR),
    )
    saved_paths = foggify.generate_fog(samples)

    # ── verify ──
    assert len(saved_paths) == len(samples), (
        f"Expected {len(samples)} outputs, got {len(saved_paths)}"
    )

    for path in saved_paths:
        path = Path(path)
        assert path.exists(), f"Output file missing: {path}"
        assert path.suffix == ".png"

        # Verify it is a valid image with the right shape
        img = np.asarray(Image.open(path))
        assert img.ndim == 3 and img.shape[2] == 3, (
            f"Expected RGB image, got shape {img.shape}"
        )
        assert img.dtype == np.uint8

    # Verify a model config.json was written alongside the images
    model_dirs = [p for p in OUTPUT_DIR.iterdir() if p.is_dir()]
    assert len(model_dirs) > 0, "No model output directories created"
    for model_dir in model_dirs:
        cfg_file = model_dir / "config.json"
        assert cfg_file.exists(), f"Missing config.json in {model_dir}"
        with open(cfg_file) as f:
            model_cfg = json.load(f)
        assert "size" in model_cfg
