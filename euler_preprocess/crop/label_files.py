"""Crop label rasters in the stored file, so ground-truth labels keep every class.

A semantic segmentation is ground truth: sky, road, the ego vehicle and every
other class are encoded in one file, which consumers read through different
functions (``class_segmentation`` for the labels, ``sky_mask`` for the sky).
Re-encoding the decoded value through a dataset writer keeps what one loader
returns, but not necessarily the file:

- palettes and bit depths change (an indexed or 16-bit PNG comes back as 8-bit);
- channels the loader ignores are dropped (Real Drive Sim keeps class IDs in
  the red channel of an RGBA PNG; its writer zeroes the others);
- a semantic map read through ``sky_mask`` would be written back as sky / not sky.

So label rasters are cropped in the file itself: the stored pixels are sliced,
and resized by picking the source pixels nearest resampling picks, in the
same image mode, palette and format. PNG, TIFF and BMP go through Pillow, and
``.npy`` / ``.npz`` through NumPy. The caller then checks that the modality's
loader reads the result as the crop of the source values. Anything else
(JPEG, 16-bit colour PNGs Pillow cannot hold, multi-frame images, ambiguous
array layouts) returns ``None`` and is re-encoded with the dataset's writer.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np

from euler_preprocess.crop.spec import CropBox

#: Formats Pillow writes back losslessly.
_LOSSLESS_IMAGE_FORMATS = {"PNG", "TIFF", "BMP"}
_IMAGE_SUFFIXES = {".png", ".tif", ".tiff", ".bmp"}
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
#: Pillow modes that hold 16-bit samples.
_WIDE_MODES = {"I", "I;16", "I;16B", "I;16L", "I;16N"}
#: TIFF compressions that lose nothing.
_LOSSLESS_TIFF_COMPRESSIONS = {None, "raw", "tiff_lzw", "tiff_adobe_deflate", "tiff_deflate", "packbits"}


def source_grid(box: CropBox) -> tuple[np.ndarray, np.ndarray]:
    """Return the source row of every output row and the source column of every output column.

    Without a resize these are the crop's rows and columns. With one they are
    the pixels euler-loading's nearest resize picks: it is separable, so
    resizing a column of row indices (and a row of column indices) gives them
    exactly.
    """
    rows = np.arange(box.top, box.top + box.height)
    cols = np.arange(box.left, box.left + box.width)
    if box.resized:
        from euler_loading import resize_raster

        pick_rows = resize_raster(
            np.arange(box.height, dtype=np.int64)[:, None],
            layout="HW",
            size=(box.output_height, 1),
            interpolation="nearest",
        )
        pick_cols = resize_raster(
            np.arange(box.width, dtype=np.int64)[None, :],
            layout="HW",
            size=(1, box.output_width),
            interpolation="nearest",
        )
        rows = rows[np.asarray(pick_rows)[:, 0]]
        cols = cols[np.asarray(pick_cols)[0, :]]
    return rows, cols


def crop_label_file(
    data: bytes, name: str, rows: np.ndarray, cols: np.ndarray, source_size: tuple[int, int]
) -> bytes | None:
    """Return the file *data* with only pixels ``rows`` x ``cols`` kept, or ``None``.

    *name* gives the format (its suffix) and *source_size* the ``(height,
    width)`` the stored raster must have. ``None`` means the file cannot be
    cropped in place without loss.
    """
    suffix = Path(name).suffix.lower()
    if suffix in _IMAGE_SUFFIXES:
        return _crop_image(data, rows, cols, source_size)
    if suffix == ".npy":
        return _crop_npy(data, rows, cols, source_size)
    if suffix == ".npz":
        return _crop_npz(data, rows, cols, source_size)
    return None


def _png_bit_depth(data: bytes) -> int | None:
    if data[:8] != _PNG_SIGNATURE or data[12:16] != b"IHDR" or len(data) < 25:
        return None
    return data[24]


def _crop_image(data: bytes, rows: np.ndarray, cols: np.ndarray, source_size: tuple[int, int]) -> bytes | None:
    from PIL import Image

    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception:
        return None
    with image:
        if image.format not in _LOSSLESS_IMAGE_FORMATS or getattr(image, "n_frames", 1) != 1:
            return None
        if (image.height, image.width) != tuple(source_size):
            return None
        if _png_bit_depth(data) == 16 and image.mode not in _WIDE_MODES:
            # Pillow reads 16-bit colour PNGs as 8-bit: the file cannot be kept.
            return None
        params: dict[str, object] = {}
        if image.format == "TIFF":
            compression = image.info.get("compression")
            if compression not in _LOSSLESS_TIFF_COMPRESSIONS:
                return None
            if compression is not None:
                params["compression"] = compression
        if "transparency" in image.info:
            params["transparency"] = image.info["transparency"]

        pixels = np.asarray(image)
        cropped = Image.fromarray(np.ascontiguousarray(pixels[rows][:, cols]))
        if image.mode == "P":
            # Pillow turns the indices into an "L" image; the palette makes it "P" again.
            cropped.putpalette(image.getpalette())
        if cropped.mode != image.mode:
            return None
        buffer = io.BytesIO()
        cropped.save(buffer, format=image.format, **params)
        return buffer.getvalue()


def _spatial_axes(shape: tuple[int, ...], source_size: tuple[int, int]) -> tuple[int, int] | None | bool:
    """Return the adjacent ``(height, width)`` axes of *shape*; ``None`` if absent, ``False`` if ambiguous."""
    height, width = source_size
    matches = [axis for axis in range(len(shape) - 1) if shape[axis] == height and shape[axis + 1] == width]
    if not matches:
        return None
    if len(matches) > 1:
        return False
    return matches[0], matches[0] + 1


def _take(array: np.ndarray, axes: tuple[int, int], rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.take(np.take(array, rows, axis=axes[0]), cols, axis=axes[1]))


def _crop_npy(data: bytes, rows: np.ndarray, cols: np.ndarray, source_size: tuple[int, int]) -> bytes | None:
    try:
        array = np.load(io.BytesIO(data), allow_pickle=False)
    except Exception:
        return None
    axes = _spatial_axes(array.shape, source_size)
    if not axes:
        return None
    buffer = io.BytesIO()
    np.save(buffer, _take(array, axes, rows, cols), allow_pickle=False)
    return buffer.getvalue()


def _crop_npz(data: bytes, rows: np.ndarray, cols: np.ndarray, source_size: tuple[int, int]) -> bytes | None:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            compressed = any(info.compress_type != zipfile.ZIP_STORED for info in archive.infolist())
        with np.load(io.BytesIO(data), allow_pickle=False) as npz:
            arrays = {key: npz[key] for key in npz.files}
    except Exception:
        return None
    cropped: dict[str, np.ndarray] = {}
    for key, array in arrays.items():
        axes = _spatial_axes(array.shape, source_size)
        if axes is False:
            return None
        # Arrays without the image's axes (class names, metadata) are kept as they are.
        cropped[key] = array if axes is None else _take(array, axes, rows, cols)
    if all(cropped[key] is arrays[key] for key in arrays):
        return None
    buffer = io.BytesIO()
    (np.savez_compressed if compressed else np.savez)(buffer, **cropped)
    return buffer.getvalue()
