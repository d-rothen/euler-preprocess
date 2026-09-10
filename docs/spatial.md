# Deterministic spatial exports (3.14.0, unreleased)

`euler-preprocess spatial --config examples/spatial-five-field.json` is the
opt-in crop/resize producer. It uses loading 2.24.0, crawler 2.11.0 and contract
0.5.0. These source versions must be installed together. No benchmark evaluation
is enabled. Fog/model/value effects are not labelled replayable spatial exports.

The example config expects five source modality datasets. Change their paths to
your sources, and supply decoded profiles, image plane, qualified calibration
selection and source revision assertions that describe those files. The producer
hashes the actual source content itself; it does not accept supplied hashes as
independently checked. Paths in CLI configs are resolved against the config file.

```python
from euler_preprocess.spatial import export_spatial
paths = export_spatial(config)
```

`preprocessing` uses the shared `SamplePreprocessor` config. `bindings` has
`fields`, `image_planes`, `revisions` (source alias to revision digest) and an
optional explicit CPU `backend` (`torch_cpu` by default). `outputs` selects
fields; each needs `path`, a distinct derived `dataset_id`, and `revision`.
Optional output options are `metadata_scope`, `resume` (directories only), and
`encoding: {"id":"npy"}` or a supported PNG codec, e.g.
`{"id":"png_depth16","scale":0.001}`. NPY preserves K/rays exactly. PNG RGB
expects normalized float32 values, masks require bool, and depth requires meters.
Clipping is rejected. Codec parameters, profiles and real dimensions are stored
alongside the bytes; source metadata cannot override them.

`spatial_variants` replaces top-level preprocessing/bindings with explicit named
`{preprocessing, bindings}` entries. Every variant is frozen in the output plan
before writing. Variants can select different qualified calibrations or concrete
geometries. A `samples` list selects source indices. Optional
`mapping: {variant: {source_full_id: output_full_id}}` supplies exact output IDs.
Otherwise a single variant preserves source IDs; multiple variants append the
explicit variant ID as a hierarchy component. Conflicting mappings are rejected.
The producer computes the complete expected output set before starting writes.
Use explicit mappings/subsets, rather than inferring relationships from filenames.

Each sample carries the actual producer receipt into `SourceBackedOutputBackend`.
`prepare_output_backend` and `prepare_output_backends` also accept explicitly
configured `materialization_outputs` for Python integrations; each output option
includes a frozen `derivation` and optional `expected_full_ids`. The dataset must
already have source-backed transform capture enabled. An arbitrary transform's
changed values will fail the receipt checks. Ordinary attributes are preserved,
while the new receipt and output encoding metadata are authoritative.

`receipt_sidecars: true` moves receipts to relative content-addressed records.
Failed encoding cannot register a receipt. Finalization verifies source-backed
replay, output bytes, expected IDs and references before publishing each logical
output dataset. There is no all-or-nothing transaction across several output
roots: each head is independently complete for its one selected modality.
Directory resume is idempotent with the same source/content/recipe/revision;
ZIP supports one owner/one finalization. Interrupted outputs do not advertise
complete metadata.

The common dataset builder now consumes `preprocessing` for ordinary pipelines
as well. Existing callable pipelines retain their APIs. Strict receipts are
created only by the explicit spatial/materialization path. Reload through
`MultiModalDataset` loads stored bytes once; explicit loading replay and typed
`CalibrationView` resolve the original geometry without evaluator heuristics.
