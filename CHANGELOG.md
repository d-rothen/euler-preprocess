# Changelog

## Unreleased

- `fog` no longer requires `semantic_segmentation`. Without a sky mask no pixel
  is sky: `from_sky` airlight falls back to white, sky-only options have no
  effect, and one warning per run says so. A configured mask is joined and
  used exactly as before. For indoor datasets such as Hypersim.
- The white `from_sky` fallback for samples without sky pixels is logged as a
  warning once per run, then at debug level, instead of once per sample and
  fog variant.
- `fog` reads the depth head's `radial_depth` (the CLI logs what it found):
  - `true`: the depth is used as the distance along the ray, and intrinsics
    never convert it again (before, a radial depth plus intrinsics was
    converted twice, up to 1.9× too far for Hypersim's tilt-shift cameras).
  - `false` without intrinsics: an error, instead of silently using planar
    depth as distance. `"planar_depth_as_distance": true` in the fog config
    accepts it with one warning.
  - Undeclared: unchanged (converted when intrinsics are present).
  - With a `function` override, the loader function's own `radial_depth`
    annotation decides, e.g. Hypersim's `depth` index read with `radial_distance`.
  - Standalone `FogTransform(...)` takes it as `depth_radial`.
- Single-sample inference (`render_fog_sample` / `render_fog_image`, used for
  preview renders) follows the same rules as a run:
  - `semantic_segmentation` and `intrinsics` are optional; without a sky mask
    no pixel is sky.
  - `depth_radial` (the depth head's `radial_depth`) decides whether the
    intrinsics convert the depth, including the error for planar depth
    without intrinsics and `planar_depth_as_distance`. Before, intrinsics
    always converted it, so a radial depth was converted twice.
- Transforms can declare `OPTIONAL_MODALITIES`: per-sample inputs that anchor
  the role inference like required ones when configured.

- `crop` crops label rasters in the stored file: semantic segmentation (the
  ground truth that also holds the sky), instance and panoptic segmentation,
  and sky masks.
  - PNG, TIFF and BMP keep their mode, palette, bit depth and every channel;
    `.npy`/`.npz` keep their dtype and other arrays.
  - The result must decode exactly as the crop of the source labels.
  - So no class is lost to a writer that re-encodes only what one loader
    function returns: Real Drive Sim's non-red channels, or every class but
    the sky for a dataset read through `sky_mask`. Label modalities no longer
    need a writer.
  - Files that cannot be kept (JPEG, 16-bit colour PNGs) are re-encoded with
    the writer, with a warning.
  - Labels can only be resized with `nearest`; another `resize.interpolation`
    for them is an error.

- `crop` supports MUSES (with euler-loading 2.35.0's MUSES writers):
  - `calib.json` holds the frame camera's `K` and every extrinsic. The
    `intrinsics` modality patches it once, and every other modality indexing
    the same file, or a byte-identical copy (`camera_extrinsics`,
    `lidar_to_gnss`), is written with the patched bytes, so the bundle holds
    one `K`. What each of them reads from the patched file must stay as it was;
    two intrinsics modalities patching one file differently are refused.
  - GNSS records (`gnss`) and `meta.json` timing (`sensor_timing`) are copied
    unchanged by default.
  - `configs/crop_muses_hood_config.json` cuts the engine hood off the
    1920×1080 frame camera images (bottom 160 rows, 1920×920 stay), with
    `configs/crop_muses_dataset_config.json` as the dataset config.
- Require euler-loading 2.35.0 for the MUSES writers.

- Add `euler-preprocess lidar-motion`, which compensates scanning-lidar points for
  the vehicle's motion during the sweep. It follows the MUSES SDK, moving each
  point to the frame camera's mid-exposure using the `gnss`, `sensor_timing`
  and `lidar_to_gnss` modalities.
- Require euler-loading 2.32.0, the first release with `muses.gnss`,
  `muses.sensor_timing`, `muses.read_lidar_to_gnss` and `muses.write_sparse_depth`.
- Add `euler-preprocess crop`, which crops RGB, depth, intrinsics and
  extrinsics coherently. The box is given as fractions or pixels, either removed
  per edge or kept with an anchor, relative to a reference raster. Rasters are
  sliced, pinhole intrinsics get `cx -= left`, `cy -= top` merged into a copy of
  their file, and lidar point clouds and extrinsics are copied unchanged. Each
  modality's treatment comes from its loader annotation, not its name. Shared
  calibration files are written once, and samples with different crops cannot
  share one. The outputs are reopened and checked against the crop.
  `configs/crop_stf_valid_region_config.json` removes SeeingThroughFog's black
  rectification border.
- `crop` phase 3:
  - `point_clouds: "prune_to_crop"` keeps only the lidar points that project
    into the output image (`z > 0`, `rint(K'·T·X)` inside), the projection
    consumers such as dmd3c-euler use;
  - `resize` (`size` or `scale`) after the crop, with `K' = S·C·K`; colour
    images are interpolated bilinearly, all other rasters take the nearest
    pixel;
  - `"output_layout": "scoped"` writes one root (directory or zip) with one
    metadata scope per modality, as dmd3c-euler's `--dataset-root` reads;
  - every encoded raster or pruned cloud is decoded again and must reproduce
    the values;
  - `euler-preprocess crop-box` prints the pixel box and the matching
    `spatial` preprocessing block.
- `crop` uses the new public euler-loading accessors (`get_loader_meta`,
  `get_hierarchical_files`) and ds-crawler's `set_modality_meta` / `path_prefix`.
  Requires euler-loading 2.33.0 and ds-crawler 2.14.0.
- `extract_intrinsics` (radial, fog) also uses a hierarchical intrinsics file
  that is not named `intrinsics` when it is the only one, such as SeeingThroughFog's
  `calib_cam_stereo_left`. Several matching files now raise instead of being
  ignored.

## 3.14.0 (unreleased)

Consume shared preprocessing config and add the opt-in spatial producer, explicit variants/mappings and captured source-backed outputs. Preserve ordinary pipeline APIs.
