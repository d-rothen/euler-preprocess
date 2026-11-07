# Coherent cropping plan

Status: **phases 1 and 3 implemented** (2026-11-07) as `euler-preprocess crop`
(plus `crop-box`), for **SeeingThroughFog** (euler-loading's `princeton_dense`
loader) with `rgb`, `sparse_depth`, `intrinsics` and `camera_extrinsics`; see §5
for results. Phase 2 (other datasets) is open. The euler-loading side shipped in
2.31.0 (STF fixes) and 2.32.0 (MUSES inputs); phase 3 uses euler-loading 2.33.0
and ds-crawler 2.14.0 (both released), which euler-preprocess now requires.
euler-view launches it with the `euler-preprocess-crop` addon. Written 2026-11-06 against
euler-preprocess 3.17.0 and euler-loading 2.30.0.

## 1. Is there a pathway today?

Not one that covers RGB, sparse depth, intrinsics and extrinsics together.

| Path | What it does | Why it doesn't fit |
|---|---|---|
| `euler-preprocess spatial` (`euler_preprocess/spatial.py`, `docs/spatial.md`) | Fixed-pixel crop/resize export with verified receipts | No percentages. Rejects passthrough fields (`euler_loading/transform_descriptors.py:174`), so lidar point clouds and extrinsics can't be exported. Writes NPY/PNG `materialized` datasets, not native formats. Needs a ~260-line binding config (`examples/spatial-five-field.json`). |
| `preprocessing` block in any dataset config (`euler_preprocess/common/dataset.py:221`) | Crops in memory before fog / sky-depth / radial run | Only the transform's own output is written, and it destroys sparse depth and extrinsics (table below). |
| `euler_loading.crop_intrinsics` / `geometry.crop_matrix` | The K maths | Building block only. |

Verified hazard (euler-loading 2.30.0,
`SamplePreprocessor.from_config({"crop": {"size": [576, 1024]}})` on one sample):

| Field | In | Out |
|---|---|---|
| `rgb` | 720×1280×3 | 576×1024×3 ✓ |
| `intrinsics` (3×3) | cx 640 | cx 512 ✓ |
| `sparse_depth` (N, 6) | 120000×6 | **576×0** |
| `camera_extrinsics` (4×4) | 4×4 | **0×0** |
| `intrinsics` as `{file_id: K}` (hierarchical) | — | **AttributeError** |

Cause: the field kind is inferred from the name (`sparse_depth` → `depth`,
`preprocessing.py:100-112`) or from a 2-D shape (4×4 → `generic`, `:447`), and
`_crop_spatial_value` (`:695`) then slices axes 0 and 1 of any 2-D array.

## 2. What a crop changes

Box `(top, left, h, w)` in RGB pixels.

| Item | Correction |
|---|---|
| RGB, dense depth, sparse-depth rasters, masks | Slice. No resampling, so values are unchanged (planar and radial depth alike). |
| Pinhole matrix (`K`, or `P` for rectified images) | `K' = C·K` with `C = [[1,0,-left],[0,1,-top],[0,0,1]]`: `cx -= left`, `cy -= top`; fx, fy and skew unchanged |
| Distortion coefficients | Unchanged |
| Image-size metadata | Update: head `modality.meta.dimensions`, ROS `width`/`height`, Synscapes `resx`/`resy`, RDS `fov` |
| Extrinsics | **Unchanged** |
| Lidar point clouds | **Unchanged** |

The extrinsics need no correction because `u' = C·u = C·K·[R|t]·X = K'·[R|t]·X`:
the whole correction lives in the pixel matrix. The official SeeingThroughFog
tooling does exactly this. `resize.get_image_scaling()` returns
`[[sx,0,-sx·left],[0,sy,-sy·top],[0,0,1]]`, which is left-multiplied onto
`P·velodyne_to_camera`, and the lidar→camera transform is left alone
(`tools/CreateTFRecords/generic_tf_tools/resize.py:173`,
`tools/ProjectionTools/Lidar2RGB/lib/visi.py:41`).

So the cropped bundle carries the extrinsics and point clouds as unchanged
copies. Consumers that project at load time (dmd3c-euler `project_sparse_depth`,
`euler_complete.py:175`, and euler-eval) then get exactly the crop of the
original projection, bounded by the cropped RGB's h×w.

Later datasets have one exception. Where a single file holds both K and the
extrinsics (Real Drive Sim and MUSES `calib.json`, Synscapes `meta/<id>.json`),
the extrinsics output must be the patched file too. Otherwise the bundle carries
two different K. SeeingThroughFog keeps them in separate files.

## 3. SeeingThroughFog

### 3.1 Inputs and outputs

| Modality | Source (paths illustrative) | Loader (`princeton_dense`) | Stored as | Crop role | Output |
|---|---|---|---|---|---|
| `rgb` | `cam_stereo_left_lut/*.png` | `rgb` (HWC float32) | 8-bit PNG, 1920×1024, rectified | raster | Cropped PNG via `write_rgb`; the round trip is bit-exact (checked) |
| `sparse_depth` | `lidar_hdl64_strongest/*.bin` (or `_last`) | `sparse_depth` (`NC`) | (N, 5) float32 `x, y, z, intensity, ring` in `lidar_hdl64_s3_roof` | point set | Byte copy |
| `intrinsics` | `calib_cam_stereo_left.json`, one file, matched hierarchically | `read_intrinsics` (3×3) | ROS CameraInfo: `K, D, R, P, width, height, roi, …` | intrinsics | Patched copy via `princeton_dense.write_intrinsics`: K and P `cx`/`cy`, `width`, `height` |
| `camera_extrinsics` | `calib_tf_tree_full.json`, one file | `read_extrinsics` (4×4, HDL64 → `cam_stereo_left_optical`) | ROS TF list | unchanged | Byte copy |

The byte copies need no writer. `princeton_dense.write_intrinsics` (euler-loading
2.31.0, §3.2) covers the intrinsics patch.

### 3.2 Fixed in euler-loading 2.31.0: `read_intrinsics` returned the wrong matrix for `_lut` images

- `princeton_dense.read_intrinsics` returns `K`
  (`euler_loading/loaders/_princeton_dense.py:170`).
- The `cam_stereo_left_lut` PNGs are rectified to `P`. The official conversion
  calls `rectifyImage` → `cv2.initUndistortRectifyMap(K, D, R, P)`
  (`tools/Raw2LUTImages/conversion_lib/process.py:115`,
  `pinhole_camera_model.py:125`). The official lidar projection uses `P`:
  `vtc = P @ R @ velodyne_to_camera` with `R = I`
  (`tools/DatasetViewer/lib/read.py:176-183`).
- The two matrices differ a lot: fx 2612.86 vs 2355.72, cx 966.53 vs 988.14.
  Projecting lidar with `K` into a `_lut` image misplaces points by −21.6 px at
  the principal point, −129 px at the left edge, +80 px at the right edge and
  about ±55 px at the top and bottom.
- This is independent of cropping, but a crop corrects whichever matrix the
  loader returns, so it had to be settled first.
- Released in euler-loading 2.31.0: `read_intrinsics` now returns the
  rectified `P[:, :3]` by default (a behaviour change, in its CHANGELOG).
  `intrinsics_matrix: "K"` (alias `"raw"`) in per-file attributes or head
  meta selects the raw K.
- `write_intrinsics` writes that matrix back into a camera-info JSON. An
  existing file is updated: the change from the stored matrix must be a crop
  or rescale, and it is applied to K and P alike (whole-pixel crops as exact
  float64 integer shifts). D, R, roi and header are kept; `meta["width"]` and
  `meta["height"]` set the size. Any other change is refused. New files and
  streams get an ideal pinhole camera.
- `K` (with `D`) belongs only to the raw 12-bit `cam_stereo_left` TIFFs (`rccb`).
- `R` (a 0.256° rotation, about 10 px) is left out, as the official tools and
  their KITTI-format calibration (`R0_rect = I`) do. On two real dense-fog
  frames (2018-10-08_08-18-59_00650 / 01050), a road-marking correlation scan
  puts the best fit for `P` at (−3, +8) px, against (+7, +13) px for `P·R`.
  The raw TIFFs agree independently: `K+D` with `Rᵀ` is closer than plain
  `K+D` (12 vs 24 px). So the TF camera frame is the rectified one. Both image
  types keep a ~+8–10 px vertical residual, which looks like lidar/camera
  timing or a small pitch error. Re-check on static, clear-weather frames.
  Plain `K` is off by 26 px and visibly misses a road sign by about 45 px.

### 3.3 Other STF pitfalls

- The `lidar_hdl64_*_stereo_left` and `*_gated` point clouds are already in a
  camera frame (official README table). Don't pair them with the default
  HDL64→camera extrinsics; use `lidar_hdl64_strongest` or `_last`.
- `rccb` (raw Bayer TIFF) has no writer. A crop would need even offsets to keep
  the Bayer phase, and demosaicing makes the 1-px border differ from cropping
  the demosaiced image. Out of scope for phase 1.
- Raw TIFF datasets (`princeton_stf+rgb`):
  - `princeton_dense.rgb` silently returns an all-white image for them: Pillow
    clips 12-bit data. Their head must name `rccb`.
  - `rccb` swapped red and blue; it now demosaics as GRBG (fixed in
    euler-loading 2.31.0).
  - The raw frames are not pinhole images (k1 = −0.57), so use the `_lut`
    images for depth evaluation.
- The official crops are pixel boxes, e.g. `RGB2Gatedv2` keeps rows 202–970 and
  columns 280–1560 (1280×768, `resize.py:91`). Accept pixel units alongside
  fractions so these reproduce exactly.
- Invalid rectification border. By the calibration, 5.1 % of each `_lut` frame
  samples outside the raw sensor: bands at the middle of the top, bottom and
  left edges. OpenCV's default `remap` fills those with black.
  `remove: {top: 34, bottom: 46, left: 16, right: 0}` (pixel units) leaves a
  fully valid 1904×944 image (91.4 % of the frame). That's a sensible default
  crop for depth evaluation, or else mask those pixels. Computed from
  `calib_cam_stereo_left.json` and the official conversion code, and confirmed
  on two real dense-fog frames: the black pixels reach 33 rows at the top, 45 at
  the bottom and 15 columns on the left, and none remain inside the box. Shipped
  as `configs/crop_stf_valid_region_config.json`.
- Resampling in the official rectification is mild but uneven: about 1.11 raw
  px per output px at the centre, but 0.76–0.81 near the left and right edges
  (≈1.3× upsampled, so softer there). The `_lut` tone mapping also applies
  CLAHE, which raises local contrast in fog. A pinhole image re-rendered from
  the raw TIFFs with the same K, D, R and P keeps the geometry (and `P`) but
  can avoid CLAHE. That's a photometric choice, not needed for geometric
  correctness.

### 3.4 Phase 1 acceptance

Run on an STF-format fixture (official calibration file layout, image size
consistent with `width`/`height`), and optionally on real frames via
`EULER_PREPROCESS_STF_ROOT`, the same opt-in pattern as the RDS test.

1. `rgb_out == rgb_in[top:top+h, left:left+w]`, bit-exact.
2. The `.bin` files and `calib_tf_tree_full.json` are byte-identical.
3. In the calibration JSON, `K[2]` and `P[2]` drop by `left` and `K[5]` and
   `P[6]` by `top`, computed in float64 (966.525 → 686.525, not 686.5250244).
   `width`/`height` become w/h. `D`, `R`, `roi`, `header` and `binning_*` are
   untouched.
4. Projecting the points with the cropped `P` and unchanged `T` into h×w equals
   the crop of the uncropped projection. Compare pixel coordinates to 1e-6 px.
   Rasters must match except for points within 1e-6 px of a rounding boundary,
   because `np.rint` rounds half to even.
5. The output roots reload through `MultiModalDataset` with the roles
   re-inferred (`sparse_depth` per sample, calibration hierarchical), and
   `dmd3c-euler/euler_complete.py` runs on them unchanged.

## 4. Design

### 4.1 Command and config

```bash
euler-preprocess crop -c configs/crop_stf_dataset_config.json
```

The dataset config keeps the existing shape:

```json
{
  "transform_config_path": "crop_config.json",
  "output_path": "/data/stf_crop",
  "modalities": {
    "rgb": "/data/stf/cam_stereo_left_lut",
    "sparse_depth": "/data/stf/lidar_hdl64_strongest",
    "intrinsics": "/data/stf/calib_cam_stereo_left",
    "camera_extrinsics": "/data/stf/calib_tf_tree"
  },
  "samples": {"start": 0, "count": 100}
}
```

Each modality is written to `<output_path>/<name>`, or wherever a
`pipeline.output_targets[]` entry with `slot: "<name>"` points (directory or
zip). A crop config containing `preprocessing` is rejected, since it would crop
twice.

`crop_config.json`:

```json
{
  "reference": "rgb",
  "crop": {"remove": {"top": 0.20, "bottom": 0.05, "left": 0.15, "right": 0.19}},
  "point_clouds": "keep",
  "verify": "all"
}
```

Other `crop` forms:

- `{"keep": {"height": 0.75, "width": 0.67}, "anchor": "bottom"}`. Anchors:
  `center`, `top`, `bottom`, `left`, `right`, `top_left`, `top_right`,
  `bottom_left`, `bottom_right`.
- `{"remove": {"top": 202, "bottom": 54, "left": 280, "right": 360}, "unit": "pixel"}`,
  which equals `RGB2Gatedv2`.
- Optional `"multiple_of": 32` trims the kept size down to a multiple (DMD3C
  pads to its stride today).

Fractions become pixels once per distinct reference size, and the result is
logged:

- `remove`: per edge, `px = floor(f·size + 0.5)`. That is round half up, not
  Python's half-to-even `round`.
- `keep`: `out = floor(k·size + 0.5)`, with offsets from the anchor (center is
  `(H - h) // 2`, as `euler_loading.Crop.resolve_box` does).
- Fractions must lie in [0, 1) and leave at least 1 px. A value like `10`, meant
  as 10 %, is an error.

The resolved box is recorded per file (`attributes.crop`: `top`, `left`,
`height`, `width`, `source_height`, `source_width`) and in each output head's
meta.

### 4.2 Roles come from the loader annotation, not the name

Every euler-loading loader carries `@modality_meta(shape=…, type=…)`, available
as `loader._modality_meta`. The crop reads that and never guesses from the
modality name, which is what causes the hazard in §1.

| Annotated shape | Role | Operation |
|---|---|---|
| `HW`, `1HW`, `HWC`, `CHW`, `3HW` | raster | Slice with the box; must match the reference H×W, else error |
| `NC` | point set | Byte copy (`prune_to_crop` later) |
| `3x3` with type `intrinsics` | intrinsics | `K' = crop_intrinsics(K, top, left)` |
| `4x4`, `Nx1` | extrinsics | Byte copy (patched copy where the file also holds K) |
| `dict` (Real Drive Sim `calibration`) | calibration map | Phase 2 |
| anything else | — | Error; override with `"roles": {"<name>": "raster" \| "copy"}` |

### 4.3 Writing

1. **Rasters**: a `SourceBackedOutputBackend` per modality, using the dataset's
   writer. The head is mirrored from the source with
   `modality.meta.dimensions` replaced (if present) and a `crop` record added.
   This is the hook radial uses for `radial_depth` (`common/output.py:685-691`),
   made independent of `transform_class`.
2. **Point sets**: a new byte-copy path. Read the source file (directory or
   zip) and register it at the same full ID and basename through
   `DatasetWriter.get_path` or `ZipDatasetWriter.write`, keeping its
   attributes. Bit-exact, and no writer needed.
3. **Hierarchical calibration** (both STF calibration files): walk the
   modality's own index (`collect_hierarchical_files`), not the samples. The
   current backend can't write these (`common/output.py:463`): hierarchical
   values arrive as `{file_id: value}` with no `sample["meta"]` entry
   (euler-loading `dataset.py:1396`), and `write_sample` refuses them
   (`dataset.py:1157`). For each file:
   - collect the boxes of the selected samples that attach to it
     (`match_hierarchical_files`);
   - require exactly one box, else raise an error naming the conflicting sizes;
   - write it once at its original full ID.

   Files that no selected sample uses are skipped.
4. **Intrinsics files**: copy the source file to the destination (a temp file
   for zip), then call the module's merge-capable
   `write_intrinsics(dest, K', meta | {"width": w, "height": h})`. It derives
   the crop from the stored matrix and rewrites only the camera fields;
   Synscapes' writer already merges this way. Before writing, check that the
   stored `width`/`height` equal the source RGB size, which catches a K
   describing another resolution.
5. **Verify** (default: every file): reopen the output roots with
   `build_dataset` and compare each sample with the cropped source. Rasters and
   K' must be equal, and point sets and extrinsics identical. On a mismatch,
   fail with the file name. This guards against writer surprises (§7).

### 4.4 Code layout (euler-preprocess)

- `euler_preprocess/crop/spec.py`: config parsing and box resolution (pure
  functions).
- `euler_preprocess/crop/roles.py`: role from the loader annotation, plus
  overrides.
- `euler_preprocess/crop/run.py`: sample loop, raster slicing (NumPy HWC or
  torch CHW, from the annotation), hierarchical bookkeeping.
- `euler_preprocess/crop/calibration.py`: hierarchical writer (unique-box
  check, copy then merge).
- `euler_preprocess/crop/verify.py`.
- `euler_preprocess/crop/files.py` (added while implementing): zip-aware reads
  of source files, value fingerprints for the verify step, and the
  recorded-image-size check for JSON calibration files.
- `euler_preprocess/common/output.py`: byte-copy registration helper, and a
  source-backed backend factory for any modality with head-meta overrides.
- `euler_preprocess/common/intrinsics.py`: `single_calibration(value)`. A
  `{id: K}` with one entry yields K; several entries are an error. Today
  `extract_intrinsics` only finds the file ID `"intrinsics"` (`:23`), so it
  misses STF's `calib_cam_stereo_left`.
- `euler_preprocess/cli.py`: a `crop` subcommand with its own `_cmd_crop`, like
  `_cmd_spatial`, because the config drives the modality set.
- `configs/crop_config.json`, `configs/crop_stf_dataset_config.json`, a README
  section and a CHANGELOG entry. The `euler-loading` pin (`>=2.32.0`, raised
  for `lidar-motion`) already covers 2.31.0.
- Optional `tools/check_crop_projection.py`: overlay projected lidar on the
  cropped RGB for a few frames, as a visual K-vs-P and crop check.

## 5. Work plan

### Phase 1: SeeingThroughFog end to end — implemented 2026-11-07

Results:
- **Tests:** `tests/test_crop.py` (STF-format fixture: directory and zip
  outputs with the pipeline manifest, sample selection, the hierarchical
  conflict, a calibration for another resolution, a raster without a writer,
  unsafe configs), `test_crop_spec.py`, `test_crop_roles.py`,
  `test_intrinsics.py`. Full suite 273 passed, 3 skipped. The opt-in real-data
  test (`EULER_PREPROCESS_STF_ROOT`) passes on two real frames.
- **Real frames** (two dense-fog `_lut` frames, valid-region crop): every §3.4
  criterion holds. RGB is the bit-exact slice, lidar and the TF tree are
  byte-identical, and `K`/`P` shift by exact float64 integers with `D`, `R`,
  `roi` and the header untouched. Projections with the cropped `P` agree with the
  shifted originals to 1e-10 px.
- **dmd3c-euler:** its own `build_dataset` loads the cropped roots unchanged, and
  its `project_sparse_depth` raster equals the crop of the uncropped projection
  (0 differing pixels). DMD3C inference itself was not run (no checkpoint or GPU
  available).

Notes from implementing:
- euler-loading's datasets resolve the GPU (torch, CHW) loader variant, so the
  crop slices CHW tensors in practice; the role's layout comes from the
  annotation of that loader.
- Each output head's `meta.crop` holds `reference`, `spec` and `boxes` (one
  box per reference size), set at finalize. `meta.dimensions` follows the crop
  when present.
- With `pipeline.output_targets`, every modality needs a target (slot = name,
  `datasetType` must match the source); the manifest lists them all and is
  written after verification passes.
- Outputs inside a source dataset are refused.

euler-loading (released in 2.31.0):

1. `princeton_dense.write_intrinsics` (CPU and GPU share one function) merges
   into an existing ROS CameraInfo JSON, as described in §3.2.
2. `princeton_dense.read_intrinsics` defaults to `P[:, :3]`, with an
   `intrinsics_matrix` selector (`"P"` | `"K"`).
3. Tests in `tests/test_princeton_dense.py`, including a lidar projection
   check that a crop needs no extrinsics change. The full suite passes.

euler-preprocess:

4. `spec.py` and tests.
5. `roles.py` and tests, including that `sparse_depth` and `camera_extrinsics`
   are never sliced.
6. Output helpers (byte copy, head-meta overrides) and tests for directory and
   zip outputs.
7. `run.py`, `calibration.py`, `verify.py` and the `crop` CLI.
8. The STF fixture test (§3.4), a hierarchical conflict test (two image sizes
   under one calibration file → error), and the optional real-data test.
9. Docs and config examples.

### Phase 2: other euler-loading datasets, when needed

| Dataset | Missing |
|---|---|
| `muses` | `write_rgb`; merge `write_intrinsics` for `calib.json` `intrinsics[sensor].K`; extrinsics output = patched `calib.json`. The lidar must be ego-motion compensated first (see below). |
| `real_drive_sim` | Merge `write_intrinsics` for the calibration JSON (`CS_FRONT` fx/fy/cx/cy; `fov` is the horizontal FOV for 2048 px, so recompute or drop it); extrinsics output = patched file; RGBA sources become RGB PNGs |
| `vkitti2` | `write_intrinsics` must keep every frame/camera row (it writes one today); `read_extrinsics` header bug (§7) |
| `synscapes` | Nothing new (merge writers exist; pass `resx`/`resy`); extrinsics output = patched meta JSON |
| `generic_dense_depth` | K lives in the head's `meta["intrinsics"]`, so patch the output head instead; PNG depth with a scale override can't be re-encoded (§7), so it needs a format-level crop |
| `generic` | Nothing |

Also in phase 2: a format-level raster crop (PNG/TIFF via Pillow, NPY/NPZ via
NumPy, preserving mode, dtype and keys) for formats whose writers are missing or
lossy.

**MUSES lidar needs motion compensation.** The sensor is not a spinning lidar.
It scans a ~124° × 25° forward window top to bottom in 84.5 ms: point
timestamps correlate −0.86 to −0.94 with elevation, and there are 5 rings.
euler-loading's `muses.sparse_depth` returns the raw points, so a moving car
smears them. Upright poles in real frames lean by 0.1–0.2 m (forward motion,
about 15–30 km/h) or about 1.5° (yaw while turning). That's the "matches at
the bottom, misses at the top" seen in projections, about 20–30 px.

The official SDK (`timbroed/MUSES`, `processing/utils.py:motion_compensate_pcd`)
moves each point to mid-exposure of the camera. It uses GNSS `gSpeed` and the
angular rates (`meta.json`, `gnss/`, `calib.json:lidar2gnss`) and writes the
same `(N, 6)` `.bin`.

Verified with the real side data:
- A port of it straightens the poles (lean 0.12–0.25 m → ≤0.06 m), and the
  projections land on the poles in both a straight-ahead and a turning frame.
- Projecting with `K` + `D` is clearly wrong, so the released frame-camera
  PNGs are already undistorted and plain `K` is right.
- Across all 2500 frames: median 29 km/h; 92 % above 10 km/h; 20 % turning
  faster than 5°/s; the camera fires 29–89 % of the way through the sweep.
- The GNSS timestamp precedes mid-exposure by up to 16.6 ms, so `meta.json`'s
  exposure times are needed.

Implemented 2026-11-07 and released (euler-loading 2.32.0; euler-preprocess
`lidar-motion`, plus the euler-view addon `euler-preprocess-lidar-motion`). The
side data became ordinary euler-loading modalities, consumed by
euler-preprocess. ds-crawler is unchanged: it only interprets paths, never file
contents.

| Modality | ds-crawler indexing | euler-loading |
|---|---|---|
| `gnss` | `gnss/` as its own dataset, same hierarchy and frame-number ID regex as `lidar`, so it joins per sample. `include_regex` `_gnss\.txt$` leaves out `dataformat.txt`. | `muses.gnss`: the u-blox record in physical units |
| `sensor_timing` | `meta.json` as a root-level file (like `calib.json`), `include_regex` `^meta\.json$` | `muses.sensor_timing` decodes it once. A new generic hook, `selects_per_sample` / `SampleContext`, gives each sample its record, matched through the `path_to_*` file names `meta.json` lists. |
| `lidar_to_gnss` | `calib.json` indexed once more | `muses.read_lidar_to_gnss` (`extrinsics.lidar2gnss`) |
| output | — | `muses.write_sparse_depth`: `(N, 6)` float64 `.bin` |

`euler-preprocess lidar-motion -c <config>` reads the four modalities per
sample. It writes a `sparse_depth` dataset mirroring the input paths, with
`meta.motion_compensated` and per-file `attributes.motion_compensation`.

Results:
- **Tests:** euler-loading 891 passed; euler-preprocess 215 passed.
- **Real data:** on the three real frames the CLI output reproduces the
  verified pole straightening.

Registered in euler-dataset-contract as `kinematics.platform.state`,
`timing.platform.acquisition` and `geometry.platform.extrinsics`.
euler-preprocess requires `euler-loading>=2.32.0`, the first release with these
loaders.

The side data stays with the local dataset: MUSES license §2.3 forbids
distributing it.

### Phase 3 — implemented 2026-11-07

- `point_clouds: "prune_to_crop"`: keeps the points with `z > 0` whose pixel
  `rint(K'·T·X)` lies in the output image, the projection dmd3c-euler's
  `project_sparse_depth` uses, with a 1e-6 px margin so a rounding boundary
  never drops a visible point. Intrinsics and extrinsics are found by role
  (`projection` names them when ambiguous). Needed a SeeingThroughFog writer:
  `princeton_dense.write_sparse_depth` (euler-loading). Per-file
  `attributes.point_clouds` records the source and kept counts.
- A single scoped output root (`"output_layout": "scoped"`): one metadata scope
  per modality, its files below `<name>/` (ds-crawler writer `path_prefix`).
  Zip roots are staged as a directory and packed once, since one zip writer
  owns an archive. dmd3c's `--dataset-root` reads it.
- Resize after the crop (`resize: {size | scale}`): `K' = S·C·K` through
  euler-loading's `crop_intrinsics` and `resize_intrinsics`. Rasters go through
  the new public `euler_loading.resize_raster`, the same code as the online
  `Resize`: bilinear for `rgb`/`rccb`, nearest otherwise (depth, masks, labels).
  Sparse-depth rasters are refused.
- `euler-preprocess crop-box -c <crop config> --size H W` prints the box and a
  `spatial` `preprocessing` block (`crop` with `size`/`offset`, then `resize`).
- Every encoded raster or pruned cloud is decoded right away with the
  modality's loader and must match (exactly, or within half an 8-bit step after
  interpolation); the verify step then compares the reloaded datasets with those
  decoded values.

The two private hooks of phase 1 became public APIs:
- euler-loading: `MultiModalDataset.get_loader`, `get_loader_meta` and
  `get_hierarchical_files`, plus `get_modality_meta(loader)`;
- ds-crawler: `DatasetWriter.set_modality_meta` (and `path_prefix`).
euler-preprocess requires euler-loading 2.33.0 and ds-crawler 2.14.0.

Results:
- **Tests:** euler-preprocess 292 passed, 3 skipped; euler-loading 1022 passed;
  ds-crawler's new writer tests pass.
- **Real STF frames** (valid-region crop):
  - pruning keeps 7.4 % of the HDL64 points;
  - dmd3c-euler's projection of the pruned clouds equals the projection of
    all points through the output camera, for the crop alone and with a
    0.5x resize (472×952).

An euler-view addon (`euler-preprocess-crop`) launches the crop for a dataset
family.

## 6. Decisions

Phase 1 adopted every recommendation below.

| # | Decision | Recommendation |
|---|---|---|
| D1 | STF matrix | Resolved: the default is now `P`, with `intrinsics_matrix: "K"` for the raw matrix |
| D2 | Crop semantics: does "percentage" mean the share removed or the share kept? | Support both (`remove` per edge, `keep` + `anchor`) plus `unit: "pixel"`, with no default guessing |
| D3 | Point clouds: keep every point or prune to the crop | Keep: lossless, and consumers already clip to the image |
| D4 | Where calibration writers live | euler-loading, next to the loaders that parse the same formats |
| D5 | Output layout: one root per modality or one scoped root | One root per modality (current convention); scoped root in phase 3 |
| D6 | Build on `spatial`, or a separate source-backed `crop` command | Separate `crop`. Extending `spatial` first needs contract support for passthrough fields. Reuse `euler_loading.crop_intrinsics` so the K maths matches. |

## 7. Pre-existing issues found while planning

1. `SamplePreprocessor` crops silently destroy point clouds and 4×4 extrinsics
   (§1). Anyone using a `preprocessing` block with STF lidar is affected. Fix in
   euler-loading: crop or resize only values whose spatial size matches the
   reference, or classify fields from the loader annotations.
2. `princeton_dense.read_intrinsics` returned raw `K` for rectified `_lut`
   images (§3.2). Fixed in euler-loading 2.31.0.
3. `generic_dense_depth` PNG depth with `scale_to_meters_override` doesn't
   survive decode → write: 10.0 m reloads as 0.039 m. The loader applies the
   scale (`generic_dense_depth.py:133`), `write_depth` (`:226`) writes
   `rint(metres)` as uint16, and the output entry keeps the override. The same
   applies to sky-depth and radial on such data.
4. `vkitti2.read_extrinsics` calls `np.loadtxt(path)` without `skiprows`
   (`vkitti2.py:208`), so it raises on the documented header line
   `frame cameraID r1,1 …`. This is also open question 4 in
   `euler-dataset-contract/docs/ecosystem-streamlining-plan.md`.
5. `extract_intrinsics` only resolved hierarchical intrinsics whose file ID is
   `"intrinsics"`, so radial and fog missed STF's `calib_cam_stereo_left`.
   Fixed with phase 1 (`single_calibration`): a single file is used whatever
   its ID; several files raise.
6. RGB writers always emit PNG bytes. VKITTI2 `.jpg` sources become PNG data
   under `.jpg` names, and Real Drive Sim RGBA sources lose their alpha channel.
7. `princeton_dense.rccb` demosaiced as GBRG, swapping red and blue. Fixed in
   euler-loading 2.31.0 (GRBG).
8. `princeton_dense.rgb` on a 12-bit TIFF returns an all-white image without
   an error. Not fixed: a guard that raises would be a small follow-up.
