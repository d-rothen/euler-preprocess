"""Label rasters (semantic segmentation and the like) are cropped in the stored file.

Ground-truth label files hold every class (sky included) and are read through
several functions, so the crop keeps the file itself: mode, palette, bit depth
and channels the loader ignores. Synthetic data only.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest
from ds_crawler import DatasetWriter
from euler_loading import Modality, MultiModalDataset, resize_raster
from PIL import Image

from euler_preprocess.cli import _cmd_crop
from euler_preprocess.crop.label_files import crop_label_file, source_grid
from euler_preprocess.crop.spec import CropBox

HEIGHT, WIDTH = 24, 40
ROWS, COLS = np.arange(3, 21), np.arange(5, 37)  # top 3, left 5, 18x32
SKY = (70, 130, 180)


def _png(image: Image.Image, **params) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", **params)
    return buffer.getvalue()


def _labels(seed: int = 0, high: int = 34) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, high, size=(HEIGHT, WIDTH)).astype(np.uint8)


def _crop(data: bytes, name: str = "x.png", rows=ROWS, cols=COLS) -> bytes | None:
    return crop_label_file(data, name, rows, cols, (HEIGHT, WIDTH))


def _png_rgb16(pixels: np.ndarray) -> bytes:
    """A 16-bit-per-channel RGB PNG, which Pillow can read (as 8-bit) but not write."""

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))

    height, width, _ = pixels.shape
    raw = b"".join(b"\x00" + pixels[row].astype(">u2").tobytes() for row in range(height))
    header = struct.pack(">IIBBBBB", width, height, 16, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


# -- the file-level crop -------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "pixels"),
    [
        ("L", _labels()),
        ("I;16", (_labels(1).astype(np.uint16) * 1000)),
        ("RGBA", np.dstack([_labels(2), _labels(3, 256), _labels(4, 256), np.full((HEIGHT, WIDTH), 255, np.uint8)])),
        ("RGB", np.dstack([_labels(5, 256), _labels(6, 256), _labels(7, 256)])),
    ],
)
def test_pngs_keep_their_mode_and_every_channel(mode, pixels):
    source = Image.fromarray(pixels)
    assert source.mode == mode
    cropped = Image.open(io.BytesIO(_crop(_png(source))))
    assert cropped.mode == mode
    np.testing.assert_array_equal(np.asarray(cropped), pixels[3:21, 5:37])


def test_indexed_png_keeps_its_palette_and_transparency():
    image = Image.fromarray(_labels())
    palette = list(np.random.default_rng(8).integers(0, 256, 768))
    image.putpalette(palette)
    assert image.mode == "P"
    cropped = Image.open(io.BytesIO(_crop(_png(image, transparency=0))))
    assert cropped.mode == "P"
    assert cropped.getpalette()[: 3 * 34] == palette[: 3 * 34]
    assert cropped.info.get("transparency") == 0
    np.testing.assert_array_equal(np.asarray(cropped), _labels()[3:21, 5:37])


def test_files_that_cannot_be_kept_are_left_to_the_writer():
    pixels = np.random.default_rng(9).integers(0, 65536, size=(HEIGHT, WIDTH, 3))
    assert Image.open(io.BytesIO(_png_rgb16(pixels))).mode == "RGB"  # read as 8 bit
    assert _crop(_png_rgb16(pixels)) is None
    assert crop_label_file(_png(Image.fromarray(_labels())), "x.png", ROWS, COLS, (HEIGHT + 1, WIDTH)) is None
    jpeg = io.BytesIO()
    Image.fromarray(_labels()).save(jpeg, format="JPEG")
    assert _crop(jpeg.getvalue(), "x.jpg") is None


def test_numpy_label_maps_are_sliced_on_their_image_axes():
    labels = _labels().astype(np.int32)
    buffer = io.BytesIO()
    np.save(buffer, labels)
    cropped = np.load(io.BytesIO(_crop(buffer.getvalue(), "x.npy")))
    assert cropped.dtype == np.int32
    np.testing.assert_array_equal(cropped, labels[3:21, 5:37])

    # Two candidate (height, width) axis pairs: left to the writer.
    square = np.zeros((HEIGHT, HEIGHT, HEIGHT), np.uint8)
    buffer = io.BytesIO()
    np.save(buffer, square)
    assert crop_label_file(buffer.getvalue(), "x.npy", ROWS, ROWS, (HEIGHT, HEIGHT)) is None

    buffer = io.BytesIO()
    np.savez_compressed(buffer, labels=labels, names=np.array(["road", "sky"]))
    with np.load(io.BytesIO(_crop(buffer.getvalue(), "x.npz"))) as npz:
        np.testing.assert_array_equal(npz["labels"], labels[3:21, 5:37])
        assert npz["names"].tolist() == ["road", "sky"]


def test_source_grid_picks_the_pixels_of_the_nearest_resize():
    box = CropBox(top=3, left=5, height=18, width=32, source_height=HEIGHT, source_width=WIDTH,
                  output_height=7, output_width=45)
    rows, cols = source_grid(box)
    index = np.arange(HEIGHT * WIDTH, dtype=np.int64).reshape(HEIGHT, WIDTH)
    expected = resize_raster(index[3:21, 5:37], layout="HW", size=(7, 45), interpolation="nearest")
    np.testing.assert_array_equal(index[rows][:, cols], expected)


# -- through euler-preprocess crop ---------------------------------------------


def _write(root: Path, loader: str, key: str, function: str, files: dict[str, bytes], **meta) -> str:
    writer = DatasetWriter(
        root, name=f"{loader}_{key}", type=key, separator=None,
        euler_loading={"loader": loader, "function": function}, meta=meta,
    )
    for frame, data in files.items():
        writer.get_path(f"/scene/{frame}", f"{frame}.png").write_bytes(data)
    writer.save_index()
    return str(root)


def _rgb_files(frames=("000", "001")) -> dict[str, bytes]:
    rng = np.random.default_rng(10)
    return {frame: _png(Image.fromarray(rng.integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8))) for frame in frames}


def _run(tmp_path: Path, modalities: dict, crop: dict) -> Path:
    (tmp_path / "crop_config.json").write_text(json.dumps(crop), encoding="utf-8")
    config = {"transform_config_path": "crop_config.json", "modalities": modalities, "output_path": str(tmp_path / "out")}
    path = tmp_path / "crop_dataset_config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    assert _cmd_crop(argparse.Namespace(config=str(path), strict=False)) == 0
    return tmp_path / "out"


REMOVE = {"crop": {"remove": {"top": 3, "bottom": 3, "left": 5, "right": 3}, "unit": "pixel"}}


def test_semantic_gt_keeps_the_channels_its_loader_ignores(tmp_path):
    """Real Drive Sim: class IDs in red; the other channels survive the crop and a resize."""
    rgba = {
        frame: np.dstack([_labels(i), _labels(10 + i, 256), _labels(20 + i, 256), np.full((HEIGHT, WIDTH), 255, np.uint8)])
        for i, frame in enumerate(("000", "001"))
    }
    modalities = {
        "rgb": _write(tmp_path / "rgb", "real_drive_sim", "rgb", "rgb", _rgb_files(), range=[0, 255]),
        "semantic_segmentation": _write(
            tmp_path / "sem", "real_drive_sim", "semantic_segmentation", "class_segmentation",
            {frame: _png(Image.fromarray(pixels)) for frame, pixels in rgba.items()}, skyclass=[29, 0, 0],
        ),
    }
    out = _run(tmp_path, modalities, {**REMOVE, "resize": {"size": [9, 50]}})
    box = CropBox(3, 5, 18, 32, HEIGHT, WIDTH, 9, 50)
    rows, cols = source_grid(box)
    for frame, pixels in rgba.items():
        cropped = Image.open(next((out / "semantic_segmentation").rglob(f"{frame}.png")))
        assert cropped.mode == "RGBA"
        np.testing.assert_array_equal(np.asarray(cropped), pixels[rows][:, cols])


def test_semantic_gt_read_as_sky_keeps_every_class(tmp_path):
    """A loader without a label function reads the GT as sky; the crop must not binarize the file."""
    palette = np.array([SKY, (128, 64, 128), (70, 70, 70), (107, 142, 35)], np.uint8)
    labels = {frame: palette[_labels(30 + i, 4)] for i, frame in enumerate(("000", "001"))}
    modalities = {
        "rgb": _write(tmp_path / "rgb", "generic_dense_depth", "rgb", "rgb", _rgb_files(), range=[0, 255]),
        "semantic_segmentation": _write(
            tmp_path / "sem", "generic_dense_depth", "semantic_segmentation", "sky_mask",
            {frame: _png(Image.fromarray(pixels)) for frame, pixels in labels.items()}, skyclass=list(SKY),
        ),
    }
    out = _run(tmp_path, modalities, REMOVE)
    for frame, pixels in labels.items():
        cropped = np.asarray(Image.open(next((out / "semantic_segmentation").rglob(f"{frame}.png"))))
        np.testing.assert_array_equal(cropped, pixels[3:21, 5:37])
        assert len(np.unique(cropped.reshape(-1, 3), axis=0)) == 4


def test_labels_cannot_be_interpolated(tmp_path):
    modalities = {
        "rgb": _write(tmp_path / "rgb", "real_drive_sim", "rgb", "rgb", _rgb_files(), range=[0, 255]),
        "semantic_segmentation": _write(
            tmp_path / "sem", "real_drive_sim", "semantic_segmentation", "class_segmentation",
            {frame: _png(Image.fromarray(np.dstack([_labels()] * 3 + [np.full((HEIGHT, WIDTH), 255, np.uint8)]))) for frame in ("000", "001")},
            skyclass=[29, 0, 0],
        ),
    }
    crop = {**REMOVE, "resize": {"scale": 0.5, "interpolation": {"semantic_segmentation": "bilinear"}}}
    with pytest.raises(ValueError, match="only be resized with 'nearest'"):
        _run(tmp_path, modalities, crop)


def test_unsupported_label_files_fall_back_to_the_writer(tmp_path, monkeypatch, caplog):
    import euler_preprocess.crop.run as run

    monkeypatch.setattr(run, "crop_label_file", lambda *args: None)
    modalities = {
        "rgb": _write(tmp_path / "rgb", "real_drive_sim", "rgb", "rgb", _rgb_files(), range=[0, 255]),
        "semantic_segmentation": _write(
            tmp_path / "sem", "real_drive_sim", "semantic_segmentation", "class_segmentation",
            {frame: _png(Image.fromarray(np.dstack([_labels(i), _labels(9, 256), _labels(9, 256), np.full((HEIGHT, WIDTH), 255, np.uint8)])))
             for i, frame in enumerate(("000", "001"))},
            skyclass=[29, 0, 0],
        ),
    }
    with caplog.at_level(logging.WARNING):
        out = _run(tmp_path, modalities, REMOVE)
    assert "re-encoding 'semantic_segmentation' with its euler-loading writer" in caplog.text
    cropped = np.asarray(Image.open(next((out / "semantic_segmentation").rglob("000.png"))))
    np.testing.assert_array_equal(cropped[..., 0], _labels(0)[3:21, 5:37])
    assert not cropped[..., 1].any()  # the writer keeps the labels, not the other channels


def test_cropped_semantic_gt_still_gives_the_sky(tmp_path):
    """Consumers such as sky-depth read sky masks through the GT's sky_mask function."""
    labels = {frame: np.where(_labels(40 + i, 3) == 0, 23, 7).astype(np.uint8) for i, frame in enumerate(("000", "001"))}
    rgb = _write(tmp_path / "rgb", "muses", "rgb", "rgb", _rgb_files(), range=[0, 255])
    semantic = _write(
        tmp_path / "sem", "muses", "semantic_segmentation", "semantic_segmentation",
        {frame: _png(Image.fromarray(pixels)) for frame, pixels in labels.items()}, skyclass=[23, 23, 23],
    )
    out = _run(tmp_path, {"rgb": rgb, "semantic_segmentation": semantic}, REMOVE)
    dataset = MultiModalDataset(
        modalities={
            "rgb": Modality(str(out / "rgb")),
            "sky": Modality(str(out / "semantic_segmentation"), function="sky_mask"),
        }
    )
    for index in range(len(dataset)):
        sample = dataset[index]
        expected = labels[sample["id"]][3:21, 5:37] == 23
        np.testing.assert_array_equal(sample["sky"].numpy()[0], expected)
