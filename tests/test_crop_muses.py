"""``euler-preprocess crop`` on MUSES-format data (synthetic; the MUSES license forbids sharing frames).

MUSES keeps the frame camera's ``K`` and every extrinsic in one ``calib.json``,
indexed as ``intrinsics``, ``camera_extrinsics`` and ``lidar_to_gnss``. The
crop patches it once and writes that patched file for all three.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from ds_crawler import DatasetWriter
from euler_loading.loaders.cpu import muses
from PIL import Image

from euler_preprocess.cli import _cmd_crop
from euler_preprocess.crop.spec import parse_crop_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

# Synthetic calibration (not MUSES values) in the layout of MUSES calib.json,
# for a frame camera of one tenth of 1920x1080.
HEIGHT, WIDTH = 108, 192
CALIB = {
    "intrinsics": {
        "rgb": {
            "K": [[104.31234567890123, 0, 95.17654321098765], [0, 104.39876543210987, 53.71357913579135], [0, 0, 1]],
            "D": [-0.25, 0.125, 0.001, -0.002, -0.03125],
        },
        "event": {
            "K": [[102.05, 0, 64.025], [0, 102.175, 36.05], [0, 0, 1]],
            "D": [-0.375, 0.1875, 0.0, 0.0, -0.0625],
        },
    },
    "extrinsics": {
        # Lidar x forward / y left / z up to optical camera axes.
        "lidar2rgb": [[0.0, -1.0, 0.0, 0.12], [0.0, 0.0, -1.0, -0.06], [1.0, 0.0, 0.0, -0.11], [0.0, 0.0, 0.0, 1.0]],
        "lidar2gnss": [[1.0, 0.0, 0.0, 0.05], [0.0, -1.0, 0.0, 0.3], [0.0, 0.0, -1.0, -0.04], [0.0, 0.0, 0.0, 1.0]],
    },
}
FRAMES = ["670857", "670890", "671007"]
SKY, ROAD, CAR = 23, 7, 26
# The engine hood fills the bottom rows of every frame (bottom 16 px here).
HOOD = 16
HOOD_CROP = {"crop": {"remove": {"bottom": HOOD}, "unit": "pixel"}}


def _writer(
    root: Path, key: str, function: str, *, scope: str | None = None, prefix: str | None = None, **meta
) -> DatasetWriter:
    return DatasetWriter(
        root,
        name=f"muses_{key}",
        type=key,
        separator=None,
        metadata_scope=scope,
        path_prefix=prefix,
        euler_loading={"loader": "muses", "function": function},
        **({"meta": meta} if meta else {}),
    )


def _image(index: int) -> np.ndarray:
    rows, cols = np.indices((HEIGHT, WIDTH))
    return np.stack([(rows * 2 + index) % 256, (cols * 3) % 256, (rows + cols) % 256], axis=-1).astype(np.uint8)


def _labels(index: int) -> np.ndarray:
    labels = np.full((HEIGHT, WIDTH), ROAD, np.uint8)
    labels[: 30 + index] = SKY
    labels[HEIGHT - HOOD :] = CAR
    return labels


def _points(index: int) -> np.ndarray:
    rng = np.random.default_rng(index)
    n = 3000
    return np.column_stack([
        rng.uniform(1.0, 60.0, n), rng.uniform(-30.0, 30.0, n), rng.uniform(-2.5, 3.0, n),
        rng.uniform(0.0, 255.0, n), rng.integers(0, 5, n), 1676367085.7 + rng.uniform(0.0, 0.085, n),
    ])


def _write_muses(root: Path) -> dict[str, object]:
    """frame_camera/, gt_semantic/, lidar/ and calib.json, indexed like a ds-crawler MUSES setup.

    ``intrinsics`` and ``camera_extrinsics`` index the root's own calib.json
    (two metadata scopes); ``lidar_to_gnss`` indexes a byte-identical copy.
    """
    rgb = _writer(root / "frame_camera", "rgb", "rgb", range=[0, 255])
    semantic = _writer(root / "gt_semantic", "semantic_segmentation", "semantic_segmentation", skyclass=[SKY] * 3)
    lidar = _writer(root / "lidar", "sparse_depth", "sparse_depth")
    for index, frame in enumerate(FRAMES):
        full_id = f"/test/fog/day/{frame}"
        name = f"REC0176_frame_{frame}"
        Image.fromarray(_image(index)).save(rgb.get_path(full_id, f"{name}_frame_camera.png"))
        Image.fromarray(_labels(index)).save(semantic.get_path(full_id, f"{name}_gt_labelIds.png"))
        muses.write_sparse_depth(str(lidar.get_path(full_id, f"{name}_lidar.bin")), _points(index))
    for writer in (rgb, semantic, lidar):
        writer.save_index()

    text = json.dumps(CALIB, indent=2) + "\n"
    for key, function in (("intrinsics", "read_intrinsics"), ("camera_extrinsics", "read_extrinsics")):
        writer = _writer(root, key, function, scope=key)
        writer.get_path("/calib", "calib.json").write_text(text, encoding="utf-8")
        writer.save_index()
    copy = _writer(root / "lidar_to_gnss", "lidar_to_gnss", "read_lidar_to_gnss")
    copy.get_path("/calib", "calib.json").write_text(text, encoding="utf-8")
    copy.save_index()
    return {
        "rgb": str(root / "frame_camera"),
        "semantic": str(root / "gt_semantic"),
        "sparse_depth": str(root / "lidar"),
        "intrinsics": {"path": str(root), "metadata_scope": "intrinsics"},
        "camera_extrinsics": {"path": str(root), "metadata_scope": "camera_extrinsics"},
        "lidar_to_gnss": str(root / "lidar_to_gnss"),
    }


def _run(tmp_path: Path, modalities: dict, crop: dict, **extra) -> Path:
    (tmp_path / "crop_config.json").write_text(json.dumps(crop), encoding="utf-8")
    config = {"transform_config_path": "crop_config.json", "modalities": modalities, **extra}
    config.setdefault("output_path", str(tmp_path / "out"))
    path = tmp_path / "crop_dataset_config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    assert _cmd_crop(argparse.Namespace(config=str(path), strict=False)) == 0
    return Path(config["output_path"])


def _frame_file(out: Path, modality: str, frame: str, suffix: str) -> Path:
    (path,) = (out / modality).rglob(f"REC0176_frame_{frame}_{suffix}")
    return path


@pytest.fixture()
def muses_root(tmp_path) -> dict[str, object]:
    return _write_muses(tmp_path / "muses")


def test_hood_crop_slices_rasters_and_keeps_the_calibration(tmp_path, muses_root):
    out = _run(tmp_path, muses_root, HOOD_CROP)

    for index, frame in enumerate(FRAMES):
        rgb = np.asarray(Image.open(_frame_file(out, "rgb", frame, "frame_camera.png")))
        np.testing.assert_array_equal(rgb, _image(index)[: HEIGHT - HOOD])
        labels = np.asarray(Image.open(_frame_file(out, "semantic", frame, "gt_labelIds.png")))
        np.testing.assert_array_equal(labels, _labels(index)[: HEIGHT - HOOD])
        assert CAR not in labels  # the hood is gone
        source = Path(str(muses_root["sparse_depth"])).rglob(f"REC0176_frame_{frame}_lidar.bin")
        assert _frame_file(out, "sparse_depth", frame, "lidar.bin").read_bytes() == next(source).read_bytes()

    # Cutting rows off the bottom keeps the principal point: every calib.json
    # output is the untouched source file.
    source = (Path(str(muses_root["lidar_to_gnss"])) / "calib.json").read_bytes()
    for name in ("intrinsics", "camera_extrinsics", "lidar_to_gnss"):
        assert (out / name / "calib.json").read_bytes() == source


def test_crop_patches_calib_json_once_for_every_modality_indexing_it(tmp_path, muses_root):
    crop = {"crop": {"remove": {"top": 6, "bottom": HOOD, "left": 10, "right": 4}, "unit": "pixel"}}
    out = _run(tmp_path, muses_root, crop)

    patched = (out / "intrinsics" / "calib.json").read_bytes()
    # The same patched file for the extrinsics (same source path) and for
    # lidar_to_gnss (a byte-identical copy): the bundle holds one K.
    assert (out / "camera_extrinsics" / "calib.json").read_bytes() == patched
    assert (out / "lidar_to_gnss" / "calib.json").read_bytes() == patched

    document = json.loads(patched)
    K, K0 = document["intrinsics"]["rgb"]["K"], CALIB["intrinsics"]["rgb"]["K"]
    assert K[0][2] == K0[0][2] - 10 and K[1][2] == K0[1][2] - 6  # exact float64 shifts
    assert K[0][0] == K0[0][0] and K[1][1] == K0[1][1]
    assert document["intrinsics"]["rgb"]["D"] == CALIB["intrinsics"]["rgb"]["D"]
    assert document["intrinsics"]["event"] == CALIB["intrinsics"]["event"]
    assert document["extrinsics"] == CALIB["extrinsics"]

    for index, frame in enumerate(FRAMES):
        rgb = np.asarray(Image.open(_frame_file(out, "rgb", frame, "frame_camera.png")))
        np.testing.assert_array_equal(rgb, _image(index)[6 : HEIGHT - HOOD, 10 : WIDTH - 4])


def test_prune_keeps_the_points_that_project_above_the_hood(tmp_path, muses_root):
    out = _run(tmp_path, muses_root, {**HOOD_CROP, "point_clouds": "prune_to_crop"})
    K = np.asarray(CALIB["intrinsics"]["rgb"]["K"])
    T = np.asarray(CALIB["extrinsics"]["lidar2rgb"])
    for index, frame in enumerate(FRAMES):
        points = _points(index)
        camera = points[:, :3] @ T[:3, :3].T + T[:3, 3]
        uv = camera @ K.T
        with np.errstate(divide="ignore", invalid="ignore"):
            u, v = np.rint(uv[:, 0] / uv[:, 2]), np.rint(uv[:, 1] / uv[:, 2])
        visible = (camera[:, 2] > 0) & (u >= 0) & (u < WIDTH) & (v >= 0) & (v < HEIGHT - HOOD)
        kept = muses.sparse_depth(str(_frame_file(out, "sparse_depth", frame, "lidar.bin")))
        np.testing.assert_array_equal(kept, points[visible])
        assert 0 < len(kept) < len(points)


def test_two_intrinsics_modalities_patching_one_file_are_refused(tmp_path, muses_root):
    root = Path(str(muses_root["lidar_to_gnss"])).parent
    event = _writer(root, "intrinsics", "read_intrinsics", scope="event_intrinsics", sensor="event")
    event.get_path("/calib", "calib.json")
    event.save_index()
    modalities = {**muses_root, "event_intrinsics": {"path": str(root), "metadata_scope": "event_intrinsics"}}
    crop = {"crop": {"remove": {"top": 6, "bottom": HOOD}, "unit": "pixel"}}
    with pytest.raises(ValueError, match="patch the same file"):
        _run(tmp_path, modalities, crop, output_path=str(tmp_path / "refused"))

    # Copying the event camera's view of the file keeps its K, and it gets the
    # file patched for the frame camera.
    out = _run(tmp_path, modalities, {**crop, "roles": {"event_intrinsics": "copy"}})
    patched = (out / "intrinsics" / "calib.json").read_bytes()
    assert (out / "event_intrinsics" / "calib.json").read_bytes() == patched
    assert json.loads(patched)["intrinsics"]["event"] == CALIB["intrinsics"]["event"]


def test_hood_config_cuts_160_rows_off_the_frame_camera():
    config = parse_crop_config(json.loads((CONFIGS / "crop_muses_hood_config.json").read_text(encoding="utf-8")))
    box = config.geometry(1080, 1920)
    assert (box.top, box.left, box.height, box.width) == (0, 0, 920, 1920)
    dataset = json.loads((CONFIGS / "crop_muses_dataset_config.json").read_text(encoding="utf-8"))
    assert dataset["transform_config_path"] == "crop_muses_hood_config.json"


def test_scoped_zip_like_the_production_muses_archive(tmp_path):
    """One zip, one metadata scope per modality; calib.json is shared by two scopes."""
    root = tmp_path / "muses"
    rgb = _writer(root, "rgb", "rgb", scope="rgb", prefix="frame_camera", range=[0, 255])
    lidar = _writer(root, "sparse_depth", "sparse_depth", scope="lidar", prefix="lidar")
    for index, frame in enumerate(FRAMES):
        full_id = f"/test/fog/day/{frame}"
        name = f"REC0176_frame_{frame}"
        Image.fromarray(_image(index)).save(rgb.get_path(full_id, f"{name}_frame_camera.png"))
        muses.write_sparse_depth(str(lidar.get_path(full_id, f"{name}_lidar.bin")), _points(index))
    rgb.save_index()
    lidar.save_index()
    for scope, key, function in (("intrinsics", "intrinsics", "read_intrinsics"), ("extrinsics", "camera_extrinsics", "read_extrinsics")):
        writer = _writer(root, key, function, scope=scope)
        writer.get_path("/calib", "calib.json").write_text(json.dumps(CALIB, indent=2) + "\n", encoding="utf-8")
        writer.save_index()
    archive = tmp_path / "muses.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(root).as_posix())

    def scoped(scope: str) -> dict[str, str]:
        return {"path": str(archive), "metadata_scope": scope}

    modalities = {
        "rgb": scoped("rgb"),
        "sparse_depth": scoped("lidar"),
        "intrinsics": scoped("intrinsics"),
        "camera_extrinsics": scoped("extrinsics"),
    }
    crop = {"crop": {"remove": {"top": 6, "bottom": HOOD}, "unit": "pixel"}, "point_clouds": "prune_to_crop"}
    out = _run(tmp_path, modalities, crop, output_layout="scoped", output_path=str(tmp_path / "cropped.zip"))
    with zipfile.ZipFile(out) as handle:
        patched = handle.read("intrinsics/calib.json")
        assert handle.read("camera_extrinsics/calib.json") == patched
        names = handle.namelist()
    assert json.loads(patched)["intrinsics"]["rgb"]["K"][1][2] == CALIB["intrinsics"]["rgb"]["K"][1][2] - 6
    assert sum(name.endswith("_frame_camera.png") for name in names) == len(FRAMES)
