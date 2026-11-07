# Changelog

## Unreleased

- Add `euler-preprocess lidar-motion`, which compensates scanning-lidar points for
  the vehicle's motion during the sweep. It follows the MUSES SDK, moving each
  point to the frame camera's mid-exposure using the `gnss`, `sensor_timing`
  and `lidar_to_gnss` modalities.
- Require euler-loading 2.32.0, the first release with `muses.gnss`,
  `muses.sensor_timing`, `muses.read_lidar_to_gnss` and `muses.write_sparse_depth`.

## 3.14.0 (unreleased)

Consume shared preprocessing config and add the opt-in spatial producer, explicit variants/mappings and captured source-backed outputs. Preserve ordinary pipeline APIs.
