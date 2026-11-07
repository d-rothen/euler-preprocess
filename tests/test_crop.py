"""End-to-end tests for ``euler-preprocess crop`` on SeeingThroughFog-format data."""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest
from ds_crawler import DatasetWriter
from euler_loading import crop_intrinsics
from euler_loading.loaders.cpu import princeton_dense
from PIL import Image

from euler_preprocess.cli import _cmd_crop
from euler_preprocess.common.dataset import build_dataset

# Values and layout of the official calib_cam_stereo_left.json shipped with the
# SeeingThroughFog tools (tools/DatasetViewer/calibs/).
CAMERA_INFO = {
    "roi": {"width": 0, "do_rectify": False, "y_offset": 0, "x_offset": 0, "height": 0},
    "D": [-0.5675749999999999, 0.0, 0.0, 0.0, 0.0],
    "P": [2355.722801, 0.0, 988.138054, 0.0, 0.0, 2355.722801, 508.051838, 0.0, 0.0, 0.0, 1.0, 0.0],
    "width": 1920,
    "K": [2612.86, 0.0, 966.525, 0.0, 2612.86, 508.297, 0.0, 0.0, 1.0],
    "height": 1024,
    "header": {"stamp": {"secs": 1517930040, "nsecs": 610230780}, "frame_id": "cam_stereo_left_optical", "seq": 259044},
    "R": [0.999993, -0.001011, -0.003531, 0.001002, 0.999996, -0.002456, 0.003533, 0.002453, 0.999991],
    "binning_y": 0,
    "binning_x": 0,
    "distortion_model": "plumb_bob",
}
# calib_tf_tree_full.json layout: lidar and camera under one body frame.
TF_TREE = [
    {
        "header": {"frame_id": "body"},
        "child_frame_id": "lidar_hdl64_s3_roof",
        "transform": {"translation": {"x": 0.0, "y": 0.0, "z": 0.0}, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}},
    },
    {
        "header": {"frame_id": "body"},
        "child_frame_id": "cam_stereo_left_optical",
        # Optical axes: x right, y down, z forward = lidar x forward.
        "transform": {"translation": {"x": 0.1, "y": -0.2, "z": 0.3}, "rotation": {"x": -0.5, "y": 0.5, "z": -0.5, "w": 0.5}},
    },
]
FRAMES = ["2018-02-03_20-48-35_00400", "2018-02-03_20-48-35_00500", "2018-02-03_20-48-35_00600"]
HEIGHT, WIDTH = 1024, 1920
# The official RGB2Gatedv2 crop: rows 202-970 and columns 280-1560.
TOP, LEFT, CROP_H, CROP_W = 202, 280, 768, 1280
RGB2GATED = {"crop": {"remove": {"top": 202, "bottom": 54, "left": 280, "right": 360}, "unit": "pixel"}}


def _writer(root: Path, key: str, function: str, **meta) -> DatasetWriter:
    return DatasetWriter(
        root,
        name=f"stf_{key}",
        type=key,
        separator=None,
        euler_loading={"loader": "princeton_dense", "function": function},
        **({"meta": meta} if meta else {}),
    )


def _image(index: int, height: int = HEIGHT, width: int = WIDTH) -> np.ndarray:
    rows, cols = np.indices((height, width))
    return np.stack([(rows + 7 * index) % 256, (cols * 3) % 256, (rows + cols) % 256], axis=-1).astype(np.uint8)


def _points(index: int) -> np.ndarray:
    rng = np.random.default_rng(index)
    n = 4000
    return np.column_stack([
        rng.uniform(4.0, 80.0, n), rng.uniform(-25.0, 25.0, n), rng.uniform(-3.0, 4.0, n),
        rng.uniform(0.0, 255.0, n), rng.integers(0, 64, n),
    ]).astype(np.float32)


def _write_stf(root: Path, *, frames=FRAMES, sizes=None, camera_info=CAMERA_INFO) -> dict[str, str]:
    rgb = _writer(root / "cam_stereo_left_lut", "rgb", "rgb", range=[0, 255])
    lidar = _writer(root / "lidar_hdl64_strongest", "sparse_depth", "sparse_depth")
    for index, frame in enumerate(frames):
        height, width = (sizes or {}).get(frame, (HEIGHT, WIDTH))
        Image.fromarray(_image(index, height, width)).save(rgb.get_path(f"/{frame}", f"{frame}.png"))
        _points(index).tofile(lidar.get_path(f"/{frame}", f"{frame}.bin"))
    rgb.save_index()
    lidar.save_index()
    calib = _writer(root / "calib_cam_stereo_left", "intrinsics", "read_intrinsics")
    calib.get_path("/calib_cam_stereo_left", "calib_cam_stereo_left.json").write_text(
        json.dumps(camera_info, indent=4), encoding="utf-8"
    )
    calib.save_index()
    tf = _writer(root / "calib_tf_tree", "camera_extrinsics", "read_extrinsics")
    tf.get_path("/calib_tf_tree_full", "calib_tf_tree_full.json").write_text(json.dumps(TF_TREE), encoding="utf-8")
    tf.save_index()
    return {
        "rgb": str(root / "cam_stereo_left_lut"),
        "sparse_depth": str(root / "lidar_hdl64_strongest"),
        "intrinsics": str(root / "calib_cam_stereo_left"),
        "camera_extrinsics": str(root / "calib_tf_tree"),
    }


def _run(tmp_path: Path, modalities: dict, crop: dict, **extra) -> Path:
    (tmp_path / "crop_config.json").write_text(json.dumps(crop), encoding="utf-8")
    config = {"transform_config_path": "crop_config.json", "modalities": modalities, **extra}
    config.setdefault("output_path", str(tmp_path / "out"))
    path = tmp_path / "crop_dataset_config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    assert _cmd_crop(argparse.Namespace(config=str(path), strict=False)) == 0
    return Path(config["output_path"])


def _project(points: np.ndarray, matrix: np.ndarray, extrinsics: np.ndarray) -> np.ndarray:
    camera = extrinsics[:3, :3] @ points[:, :3].T.astype(np.float64) + extrinsics[:3, 3:4]
    camera = camera[:, camera[2] > 0.5]
    uv = matrix @ camera
    return (uv[:2] / uv[2]).T


@pytest.fixture()
def stf(tmp_path) -> dict[str, str]:
    return _write_stf(tmp_path / "stf")


def test_crop_writes_a_coherent_stf_bundle(tmp_path, stf):
    out = _run(tmp_path, stf, RGB2GATED)

    # 1. RGB: the bit-exact slice of the source pixels.
    for index, frame in enumerate(FRAMES):
        cropped = np.asarray(Image.open(out / "rgb" / f"{frame}.png"))
        np.testing.assert_array_equal(cropped, _image(index)[TOP:TOP + CROP_H, LEFT:LEFT + CROP_W])

    # 2. Lidar and extrinsics: unchanged copies.
    for frame in FRAMES:
        source = Path(stf["sparse_depth"]) / f"{frame}.bin"
        assert (out / "sparse_depth" / f"{frame}.bin").read_bytes() == source.read_bytes()
    tf_name = "calib_tf_tree_full.json"
    assert (out / "camera_extrinsics" / tf_name).read_bytes() == (Path(stf["camera_extrinsics"]) / tf_name).read_bytes()

    # 3. Calibration: K and P shifted by exact float64 offsets, everything else kept.
    document = json.loads((out / "intrinsics" / "calib_cam_stereo_left.json").read_text())
    assert document["K"][2] == 966.525 - LEFT and document["P"][2] == 988.138054 - LEFT
    assert document["K"][5] == 508.297 - TOP and document["P"][6] == 508.051838 - TOP
    assert (document["width"], document["height"]) == (CROP_W, CROP_H)
    for key in ("D", "R", "roi", "header", "binning_x", "binning_y", "distortion_model"):
        assert document[key] == CAMERA_INFO[key]
    unchanged_k = [i for i in range(9) if i not in (2, 5)]
    assert [document["K"][i] for i in unchanged_k] == [CAMERA_INFO["K"][i] for i in unchanged_k]

    # 4. Projecting with the cropped P and the unchanged extrinsics is the crop
    #    of the uncropped projection.
    extrinsics = princeton_dense.read_extrinsics(str(out / "camera_extrinsics" / tf_name)).astype(np.float64)
    full_p = np.asarray(CAMERA_INFO["P"]).reshape(3, 4)[:, :3]
    cropped_p = np.asarray(document["P"]).reshape(3, 4)[:, :3]
    points = princeton_dense.sparse_depth(str(out / "sparse_depth" / f"{FRAMES[0]}.bin"))
    full_uv = _project(points, full_p, extrinsics)
    cropped_uv = _project(points, cropped_p, extrinsics)
    np.testing.assert_allclose(cropped_uv, full_uv - [LEFT, TOP], rtol=0, atol=1e-6)
    off_boundary = np.all(np.abs(np.abs(cropped_uv % 1.0) - 0.5) > 1e-6, axis=1)
    np.testing.assert_array_equal(
        np.rint(cropped_uv[off_boundary]), np.rint(full_uv[off_boundary]) - [LEFT, TOP]
    )

    # 5. The outputs reload with the roles re-inferred, and K is the cropped P.
    dataset = build_dataset({"modalities": {name: str(out / name) for name in stf}}, {"rgb"})
    assert sorted(dataset.modality_paths()) == ["rgb", "sparse_depth"]
    assert sorted(dataset.hierarchical_modality_paths()) == ["camera_extrinsics", "intrinsics"]
    sample = dataset[0]
    # euler-loading returns HWC arrays or CHW tensors, depending on its loader variant.
    assert tuple(sample["rgb"].shape) in {(CROP_H, CROP_W, 3), (3, CROP_H, CROP_W)}
    expected_k = crop_intrinsics(full_p.astype(np.float32), top=TOP, left=LEFT)
    np.testing.assert_allclose(sample["intrinsics"]["calib_cam_stereo_left"], expected_k, atol=1e-3)

    # The crop is recorded in every head and on every file.
    box = {
        "top": TOP, "left": LEFT, "height": CROP_H, "width": CROP_W,
        "source_height": HEIGHT, "source_width": WIDTH, "output_height": CROP_H, "output_width": CROP_W,
    }
    for name in stf:
        meta = dataset.get_modality_index(name)["head"]["modality"]["meta"]
        assert meta["crop"]["boxes"] == [box]
        assert meta["crop"]["spec"] == {"crop": RGB2GATED["crop"], "point_clouds": "keep"}
    assert sample["attributes"]["rgb"]["crop"] == box
    assert sample["attributes"]["sparse_depth"]["crop"] == box


def test_crop_writes_zip_targets_and_the_pipeline_manifest(tmp_path, stf):
    out_root = tmp_path / "pipeline_out"
    targets = [
        {"slot": "rgb", "datasetType": "rgb", "relativePath": "rgb.zip", "path": str(out_root / "rgb.zip"), "storage": "zip"},
        {"slot": "sparse_depth", "datasetType": "sparse_depth", "relativePath": "sparse_depth.zip", "path": str(out_root / "sparse_depth.zip"), "storage": "zip"},
        {"slot": "intrinsics", "datasetType": "intrinsics", "relativePath": "intrinsics.zip", "path": str(out_root / "intrinsics.zip"), "storage": "zip"},
        {"slot": "camera_extrinsics", "datasetType": "camera_extrinsics", "relativePath": "camera_extrinsics", "path": str(out_root / "camera_extrinsics"), "storage": "directory"},
    ]
    manifest = out_root / ".euler_pipeline" / "pipeline_outputs.json"
    _run(
        tmp_path,
        stf,
        {"crop": {"keep": {"height": 0.75, "width": 0.5}, "anchor": "bottom"}},
        output_path=str(out_root),
        pipeline={"output_root": str(out_root), "outputs_manifest_path": str(manifest), "output_targets": targets},
    )

    with zipfile.ZipFile(out_root / "intrinsics.zip") as archive:
        document = json.loads(archive.read("calib_cam_stereo_left.json"))
    # keep 768 x 960 rows/columns, bottom-anchored: top 256, left 480.
    assert document["P"][2] == 988.138054 - 480 and document["P"][6] == 508.051838 - 256
    assert (document["width"], document["height"]) == (960, 768)
    assert document["D"] == CAMERA_INFO["D"]

    with zipfile.ZipFile(out_root / "sparse_depth.zip") as archive:
        assert archive.read(f"{FRAMES[1]}.bin") == (Path(stf["sparse_depth"]) / f"{FRAMES[1]}.bin").read_bytes()

    listed = json.loads(manifest.read_text())["outputs"]
    assert [(entry["slot"], entry["datasetType"], entry["storage"]) for entry in listed] == [
        ("rgb", "rgb", "zip"),
        ("sparse_depth", "sparse_depth", "zip"),
        ("intrinsics", "intrinsics", "zip"),
        ("camera_extrinsics", "camera_extrinsics", "directory"),
    ]


def test_only_selected_samples_are_cropped(tmp_path, stf):
    out = _run(tmp_path, stf, RGB2GATED, samples={"start": 1, "count": 1})
    assert sorted(p.name for p in (out / "rgb").glob("*.png")) == [f"{FRAMES[1]}.png"]
    assert sorted(p.name for p in (out / "sparse_depth").glob("*.bin")) == [f"{FRAMES[1]}.bin"]
    assert (out / "intrinsics" / "calib_cam_stereo_left.json").is_file()


def test_one_calibration_file_cannot_serve_two_image_sizes(tmp_path):
    stf = _write_stf(tmp_path / "stf", sizes={FRAMES[2]: (512, 960)})
    with pytest.raises(ValueError, match="shared by samples with different crops"):
        _run(tmp_path, stf, {"crop": {"remove": {"top": 0.25}}})


def test_calibration_for_another_resolution_is_refused(tmp_path):
    stf = _write_stf(tmp_path / "stf", camera_info={**CAMERA_INFO, "width": 960, "height": 512})
    with pytest.raises(ValueError, match="describes a 960x512 image"):
        _run(tmp_path, stf, RGB2GATED)


def test_unsafe_configs_are_refused(tmp_path, stf):
    with pytest.raises(ValueError, match="would crop twice"):
        _run(tmp_path, stf, RGB2GATED, preprocessing={"crop": {"size": [10, 10]}})
    with pytest.raises(ValueError, match="lies inside the source dataset"):
        _run(tmp_path, stf, RGB2GATED, output_path=str(Path(stf["rgb"]) / "cropped"))
    with pytest.raises(ValueError, match="reference modality 'depth' is not in the dataset config"):
        _run(tmp_path, stf, {**RGB2GATED, "reference": "depth"})
    with pytest.raises(ValueError, match="must be a per-sample raster"):
        _run(tmp_path, stf, {**RGB2GATED, "reference": "sparse_depth"})


def test_rasters_without_a_writer_are_refused_before_writing(tmp_path, stf):
    """The raw 12-bit ``rccb`` TIFFs have no euler-loading writer."""
    raw = _writer(tmp_path / "stf" / "cam_stereo_left", "rccb", "rccb")
    for frame in FRAMES:
        raw.get_path(f"/{frame}", f"{frame}.tiff").write_bytes(b"not decoded before the check")
    raw.save_index()
    with pytest.raises(ValueError, match="'rccb' \\(raster\\) has no euler-loading writer"):
        _run(tmp_path, {**stf, "rccb": str(tmp_path / "stf" / "cam_stereo_left")}, RGB2GATED)
    assert not (tmp_path / "out").exists()


STF_ROOT = os.environ.get("EULER_PREPROCESS_STF_ROOT")


@pytest.mark.skipif(not STF_ROOT, reason="set EULER_PREPROCESS_STF_ROOT to a SeeingThroughFog root")
def test_real_stf_frames_crop_to_the_valid_rectified_region(tmp_path):
    """Opt-in check on real ``_lut`` frames.

    ``EULER_PREPROCESS_STF_ROOT`` holds ds-crawler-indexed ``rgb``,
    ``sparse_depth``, ``intrinsics`` and ``extrinsics`` roots. The crop removes
    the black border the rectification leaves (top 34, bottom 46, left 16 px).
    """
    root = Path(STF_ROOT)
    modalities = {
        "rgb": str(root / "rgb"),
        "sparse_depth": str(root / "sparse_depth"),
        "intrinsics": str(root / "intrinsics"),
        "camera_extrinsics": str(root / "extrinsics"),
    }
    crop = {"crop": {"remove": {"top": 34, "bottom": 46, "left": 16}, "unit": "pixel"}}
    out = _run(tmp_path, modalities, crop, samples={"count": 3})
    written = sorted((out / "rgb").rglob("*.png"))
    assert written
    for path in written:
        source = np.asarray(Image.open(root / "rgb" / path.relative_to(out / "rgb")).convert("RGB"))
        image = np.asarray(Image.open(path).convert("RGB"))
        assert image.shape[:2] == (944, 1904)
        np.testing.assert_array_equal(image, source[34:978, 16:1920])
    source_calib = json.loads(next((root / "intrinsics").glob("*.json")).read_text())
    calib = json.loads(next((out / "intrinsics").glob("*.json")).read_text())
    assert calib["P"][2] == source_calib["P"][2] - 16 and calib["P"][6] == source_calib["P"][6] - 34


def _consumer_visible(points: np.ndarray, matrix: np.ndarray, extrinsics: np.ndarray, height: int, width: int) -> np.ndarray:
    """dmd3c-euler's ``project_sparse_depth`` visibility: z > 0 and rint(u, v) inside the image."""
    camera = points[:, :3].astype(np.float64) @ extrinsics[:3, :3].T + extrinsics[:3, 3]
    z = camera[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uvw = camera @ matrix.T
        u, v = np.rint(uvw[:, 0] / uvw[:, 2]), np.rint(uvw[:, 1] / uvw[:, 2])
    return (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)


def test_prune_keeps_exactly_the_points_that_project_into_the_crop(tmp_path, stf):
    out = _run(tmp_path, stf, {**RGB2GATED, "point_clouds": "prune_to_crop"})
    tf_name = "calib_tf_tree_full.json"
    extrinsics = princeton_dense.read_extrinsics(str(out / "camera_extrinsics" / tf_name)).astype(np.float64)
    cropped_k = princeton_dense.read_intrinsics(str(out / "intrinsics" / "calib_cam_stereo_left.json")).astype(np.float64)

    dataset = build_dataset({"modalities": {name: str(out / name) for name in stf}}, {"rgb"})
    for index, frame in enumerate(FRAMES):
        source = _points(index)
        kept = princeton_dense.sparse_depth(str(out / "sparse_depth" / f"{frame}.bin"))
        visible = _consumer_visible(source, cropped_k, extrinsics, CROP_H, CROP_W)
        assert 0 < len(kept) < len(source)
        np.testing.assert_array_equal(kept, source[visible])
    sample = dataset[0]
    assert sample["attributes"]["sparse_depth"]["point_clouds"] == {
        "mode": "prune_to_crop",
        "source_points": len(_points(0)),
        "kept_points": int(_consumer_visible(_points(0), cropped_k, extrinsics, CROP_H, CROP_W).sum()),
    }
    meta = dataset.get_modality_index("sparse_depth")["head"]["modality"]["meta"]
    assert meta["crop"]["projection"] == {"intrinsics": "intrinsics", "extrinsics": "camera_extrinsics"}
    assert meta["crop"]["spec"]["point_clouds"] == "prune_to_crop"


def test_prune_needs_a_point_cloud(tmp_path, stf):
    modalities = {name: path for name, path in stf.items() if name != "sparse_depth"}
    with pytest.raises(ValueError, match="needs a point-cloud modality"):
        _run(tmp_path, modalities, {**RGB2GATED, "point_clouds": "prune_to_crop"})


def test_resize_scales_rasters_and_k_like_the_projection(tmp_path, stf):
    from euler_loading import resize_raster

    out = _run(tmp_path, stf, {**RGB2GATED, "resize": {"scale": 0.5}})
    out_h, out_w = CROP_H // 2, CROP_W // 2

    for index, frame in enumerate(FRAMES[:1]):
        written = np.asarray(Image.open(out / "rgb" / f"{frame}.png")).astype(np.float64) / 255.0
        source = _image(index).astype(np.float32)[TOP:TOP + CROP_H, LEFT:LEFT + CROP_W] / 255.0
        expected = resize_raster(source, layout="HWC", size=(out_h, out_w), interpolation="bilinear")
        assert written.shape == (out_h, out_w, 3)
        assert np.abs(written - expected).max() <= 0.5 / 255 + 1e-6

    document = json.loads((out / "intrinsics" / "calib_cam_stereo_left.json").read_text())
    assert (document["width"], document["height"]) == (out_w, out_h)
    full_p = np.asarray(CAMERA_INFO["P"]).reshape(3, 4)
    resized_p = np.asarray(document["P"]).reshape(3, 4)
    scale_crop = np.array([[0.5, 0, 0.5 * -LEFT - 0.25], [0, 0.5, 0.5 * -TOP - 0.25], [0, 0, 1]])
    np.testing.assert_allclose(resized_p, scale_crop @ full_p, atol=1e-3)
    assert document["D"] == CAMERA_INFO["D"] and document["R"] == CAMERA_INFO["R"]

    extrinsics = princeton_dense.read_extrinsics(str(out / "camera_extrinsics" / "calib_tf_tree_full.json")).astype(np.float64)
    points = _points(0)
    full_uv = _project(points, full_p[:, :3], extrinsics)
    resized_uv = _project(points, resized_p[:, :3], extrinsics)
    expected_uv = (np.c_[full_uv, np.ones(len(full_uv))] @ scale_crop.T)[:, :2]
    np.testing.assert_allclose(resized_uv, expected_uv, rtol=0, atol=1e-3)


def test_scoped_layout_writes_one_root_like_dmd3c_dataset_root(tmp_path, stf):
    from euler_loading import Modality, MultiModalDataset

    root = tmp_path / "bundle"
    _run(tmp_path, stf, RGB2GATED, output_path=str(root), output_layout="scoped")
    scopes = json.loads((root / ".ds_crawler" / "scopes.json").read_text())["scopes"]
    assert sorted(scopes) == sorted(stf)
    assert (root / "rgb" / f"{FRAMES[0]}.png").is_file()
    assert (root / "sparse_depth" / f"{FRAMES[0]}.bin").is_file()
    assert (root / "intrinsics" / "calib_cam_stereo_left.json").is_file()

    # dmd3c-euler --dataset-root: one path, the scope found from the modality type.
    dataset = MultiModalDataset(
        modalities={
            "rgb": Modality(str(root), modality_type="rgb"),
            "sparse_depth": Modality(str(root), modality_type="sparse_depth"),
        },
        hierarchical_modalities={
            "intrinsics": Modality(str(root), modality_type="intrinsics", collapse_single=True),
            "camera_extrinsics": Modality(str(root), modality_type="camera_extrinsics", collapse_single=True),
        },
    )
    assert len(dataset) == len(FRAMES)
    assert tuple(dataset[0]["intrinsics"].shape) == (3, 3)


def test_scoped_layout_into_a_zip(tmp_path, stf):
    archive = tmp_path / "bundle.zip"
    _run(tmp_path, stf, RGB2GATED, output_path=str(archive), output_layout="scoped")
    with zipfile.ZipFile(archive) as handle:
        names = set(handle.namelist())
    assert f"rgb/{FRAMES[0]}.png" in names and ".ds_crawler/scopes.json" in names
    assert not list(tmp_path.glob(".bundle.zip.*")), "the staging directory is removed"
    dataset = build_dataset({"modalities": {name: {"path": str(archive), "metadata_scope": name} for name in stf}}, {"rgb"})
    assert len(dataset) == len(FRAMES)


def test_scoped_layout_cannot_use_pipeline_targets(tmp_path, stf):
    target = {"slot": "rgb", "datasetType": "rgb", "relativePath": "rgb", "path": str(tmp_path / "x"), "storage": "directory"}
    with pytest.raises(ValueError, match="cannot use pipeline.output_targets"):
        _run(tmp_path, stf, RGB2GATED, output_layout="scoped", pipeline={"output_targets": [target]})


def test_crop_box_prints_the_spatial_preprocessing_block(tmp_path, capsys):
    from euler_preprocess.cli import _cmd_crop_box

    path = tmp_path / "crop_config.json"
    path.write_text(json.dumps({**RGB2GATED, "resize": {"scale": 0.5}}), encoding="utf-8")
    assert _cmd_crop_box(argparse.Namespace(config=str(path), size=[HEIGHT, WIDTH])) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["box"]["output_height"] == CROP_H // 2
    assert printed["preprocessing"]["operations"] == [
        {"type": "crop", "size": [CROP_H, CROP_W], "offset": [TOP, LEFT]},
        {"type": "resize", "size": [CROP_H // 2, CROP_W // 2]},
    ]
