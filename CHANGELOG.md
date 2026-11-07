# Changelog

## Unreleased

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
