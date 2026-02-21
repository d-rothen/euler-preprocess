"""Integration test for the fog generation pipeline using euler-loading.

Run with:

    pytest tests/test_foggify_integration.py -m integration
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from euler_loading import Modality, MultiModalDataset
from euler_preprocess.fog.foggify import Foggify

# ── Modality root paths ──────────────────────────────────────────────────────

RGB_PATH = "/Volumes/Volume/Datasets/vkitti2/vkitti_2.0.3_rgb"
DEPTH_PATH = "/Volumes/Volume/Datasets/vkitti2/vkitti_2.0.3_depth"
SKY_MASK_PATH = "/Volumes/Volume/Datasets/vkitti2/vkitti_2.0.3_classSegmentation"

# ── Constants ─────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FOG_CONFIG_PATH = PROJECT_ROOT / "euler_preprocess" / "fog" / "example_config.json"
OUTPUT_DIR = PROJECT_ROOT / ".tests"

MAX_SAMPLES = 5  # limit for practical test runtime


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_cpu_fog_config(src: Path, dst: Path) -> Path:
    """Copy the fog config, forcing device to 'cpu' for test portability."""
    with open(src, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["device"] = "cpu"
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    return dst


def _build_dataset() -> MultiModalDataset:
    """Build a MultiModalDataset for VKITTI2.

    Loaders are resolved automatically by euler-loading from the ds-crawler
    index at each modality path.
    """
    return MultiModalDataset(
        modalities={
            "rgb": Modality(RGB_PATH),
            "depth": Modality(DEPTH_PATH),
            "sky_mask": Modality(SKY_MASK_PATH),
        },
    )


# ── Test ──────────────────────────────────────────────────────────────────────

@pytest.mark.integration
def test_foggify_with_real_data(tmp_path):
    """Run Foggify on real data and verify the outputs land in .tests/."""
    # ── build dataset via euler-loading ──
    dataset = _build_dataset()
    assert len(dataset) > 0, (
        f"No matching files found across modalities:\n"
        f"  rgb:      {RGB_PATH}\n"
        f"  depth:    {DEPTH_PATH}\n"
        f"  sky_mask: {SKY_MASK_PATH}"
    )

    n_samples = min(MAX_SAMPLES, len(dataset))

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
    saved_paths = foggify.generate_fog(dataset[i] for i in range(n_samples))

    # ── verify ──
    assert len(saved_paths) == n_samples, (
        f"Expected {n_samples} outputs, got {len(saved_paths)}"
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
