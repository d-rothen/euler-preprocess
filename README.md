# euler-preprocess

Physics-based preprocessing transforms for multi-modal RGB+depth datasets. Built on top of [euler-loading](https://github.com/d-rothen/euler-loading) and [ds-crawler](https://github.com/d-rothen/ds-crawler).

Available transforms:

| Command | Description |
|---|---|
| `euler-preprocess fog` | Synthetic fog via the Koschmieder atmospheric scattering model |
| `euler-preprocess sky-depth` | Override depth values in sky regions with a constant |
| `euler-preprocess radial` | Convert planar (z-buffer) depth to radial (Euclidean) depth |

## Installation

```bash
uv pip install "euler-preprocess[gpu,progress] @ git+https://github.com/d-rothen/euler-fog"
```

## Usage

```bash
euler-preprocess fog       -c configs/example_dataset_config.json
euler-preprocess sky-depth -c configs/sky_depth_dataset_config.json
euler-preprocess radial    -c configs/radial_dataset_config.json
```

## Configuration

Every subcommand takes a **dataset config** JSON that points to the input data and a **transform config**. Each modality path must be a directory indexed by [ds-crawler](https://github.com/d-rothen/ds-crawler) with an `euler_loading` property that specifies the loader and function. This allows euler-loading to auto-select the correct dataset-specific loader.

### Dataset Config

```json
{
  "transform_config_path": "configs/run1.json",
  "output_path": "/path/to/output",
  "modalities": {
    "rgb": {"path": "/path/to/rgb", "split": "train"},
    "depth": "/path/to/depth",
    "semantic_segmentation": "/path/to/classSegmentation"
  },
  "hierarchical_modalities": {
    "intrinsics": {"path": "/path/to/intrinsics"}
  }
}
```

| Field | Description |
|---|---|
| `transform_config_path` | Path to the transform-specific config (see below). `fog_config_path` is also accepted for backward compatibility. |
| `output_path` | Directory where outputs are written. |
| `modalities` | Regular modalities that participate in sample-ID intersection. Each value is either a plain path string or an object with a `path` key and an optional `split` key (see below). Which modalities are required depends on the transform (see table below). |
| `hierarchical_modalities` | Per-scene data (e.g. intrinsics). Same format as `modalities`. Loaded once per scene and cached. |

#### Inline splits

When a modality directory contains [ds-crawler](https://github.com/d-rothen/ds-crawler) split files (`.ds_crawler/split_<name>.json`), you can select a subset of the data by setting the `split` key on that modality. Sample IDs are matched by intersection across all modalities, so specifying a split on a single modality is sufficient to restrict the entire dataset.

**Required modalities per transform:**

| Transform | `modalities` | `hierarchical_modalities` |
|---|---|---|
| `fog` | `rgb`, `depth`, `semantic_segmentation` | — (intrinsics optional) |
| `sky-depth` | `depth`, `semantic_segmentation` | — |
| `radial` | `depth` | `intrinsics` |

---

## Fog Transform

### Fog Config

Controls the fog simulation.

```json
{
  "airlight": "from_sky",
  "seed": 1337,
  "depth_scale": 1.0,
  "resize_depth": true,
  "contrast_threshold": 0.05,
  "device": "cpu",
  "gpu_batch_size": 4,
  "selection": { ... },
  "models": { ... }
}
```

| Field | Description |
|---|---|
| `airlight` | **Required.** Airlight estimation method: `"from_sky"` (mean sky colour), `"dcp"` (dark channel prior), or `"dcp_heuristic"` (robust DCP with sky-guided colouring when sky pixels exist). |
| `seed` | Random seed for reproducibility. `null` for non-deterministic. |
| `depth_scale` | Multiplier applied to depth values after loading. |
| `resize_depth` | Resize the depth map to match the RGB resolution (bilinear). |
| `contrast_threshold` | Threshold *C_t* used in the visibility-to-attenuation conversion (default `0.05`). |
| `device` | `"cpu"`, `"cuda"`, `"mps"`, or `"gpu"` (alias for cuda). |
| `gpu_batch_size` | Batch size when running on GPU. Uniform-model samples are batched; heterogeneous samples are processed individually. |

### Fog Model

The core equation is the **Koschmieder model** (atmospheric scattering):

```
I_fog(x) = I(x) * t(x)  +  L_s * (1 - t(x))
```

where:

- **I(x)** is the original RGB colour at pixel *x*
- **t(x) = exp(-k * d(x))** is the transmittance, which falls exponentially with depth *d* and attenuation coefficient *k*
- **L_s** is the atmospheric light (airlight), i.e. the colour of the fog/sky light scattered towards the camera
- **k** is derived from a meteorological visibility distance *V*: `k = -ln(C_t) / V`

Distant objects are attenuated more (`t` approaches 0) and replaced by airlight, just as in real fog.

### How Each Modality is Used

**RGB** — The clean scene image. Normalised to float32 in [0, 1]. This is the *I(x)* term in the fog equation -- it gets blended with the airlight according to transmittance.

**Depth** — A per-pixel depth map in **metres**. Provides the *d(x)* term in the transmittance calculation `t(x) = exp(-k * d(x))`. Pixels with greater depth receive more fog. Invalid values (NaN, inf, negative) are clamped to zero (treated as infinitely close, receiving no fog).

**Semantic Segmentation** — A per-pixel semantic segmentation map from which a boolean sky mask is derived, loaded via euler-loading's dataset-specific `semantic_segmentation` loader. The sky mask is used for airlight estimation when the `airlight` method is `"from_sky"`: the mean RGB of all sky pixels in the clean image is used as the airlight colour *L_s*.

**Intrinsics** *(optional)* — When present, planar (z-buffer) depth is converted to radial (Euclidean) depth before fog is applied.

### Airlight Estimation

The `airlight` config key selects how the atmospheric light *L_s* is estimated:

| Method | Description |
|---|---|
| `from_sky` | Mean RGB of sky pixels in the clean image. Falls back to white `[1, 1, 1]` when no sky pixels exist. |
| `dcp` | Dark Channel Prior — selects the brightest pixel (by channel sum) among the top 0.1% darkest-channel pixels. |
| `dcp_heuristic` | Robust DCP heuristic — pools the brighter half of the top 0.1% darkest-channel pixels, and when sky pixels exist it uses the brightest sky colours as the chromaticity prior while preserving DCP-derived luminance. Optional bias controls can nudge the result toward white or a cool fog tint. |

GPU-native implementations (`DCPAirlightTorch`, `DCPHeuristicAirlightTorch`) are used automatically when running on GPU.

When `airlight` is `"dcp_heuristic"`, you can optionally add:

```json
"dcp_heuristic": {
  "patch_size": 15,
  "top_percent": 0.001,
  "white_bias": 0.1,
  "cool_bias": 0.15,
  "cool_target": [0.93, 0.97, 1.0]
}
```

- `white_bias` mixes the final airlight toward neutral white.
- `cool_bias` mixes the final airlight toward a sky-relative cool target.
- `cool_target` is the cool-white anchor used to derive that sky-relative target. When sky pixels exist, the effective cool target is a blend of the estimated sky colour and `cool_target`; without sky pixels, it falls back to the airlight estimate and `cool_target`.
- `white_bias + cool_bias` must be `<= 1`.
- The tint bias preserves the estimated airlight luminance, so it shifts colour without silently changing fog density.

### Model Selection

Each image is assigned a fog model via the `selection` block:

```json
"selection": {
  "mode": "weighted",
  "weights": {
    "uniform": 1.0,
    "heterogeneous_k": 0.0,
    "heterogeneous_ls": 0.0,
    "heterogeneous_k_ls": 0.0
  }
}
```

- **`fixed`** mode: always use a single named model.
- **`weighted`** mode: randomly select a model per image according to normalised weights.

Four models are available:

| Model | Description |
|---|---|
| `uniform` | Constant *k* and *L_s*. Standard homogeneous fog. |
| `heterogeneous_k` | Spatially-varying *k*, constant *L_s*. Simulates patchy fog / fog banks. |
| `heterogeneous_ls` | Constant *k*, spatially-varying *L_s*. Simulates scattered-light colour variation. |
| `heterogeneous_k_ls` | Both *k* and *L_s* vary spatially. Most expressive model. |

### Visibility Distribution

Each model specifies a `visibility_m` distribution from which a visibility distance (in metres) is sampled per image:

| `dist` | Parameters | Description |
|---|---|---|
| `constant` | `value` | Fixed value. |
| `uniform` | `min`, `max` | Uniform random in range. |
| `normal` | `mean`, `std`, optional `min`/`max` | Gaussian, optionally clamped. |
| `lognormal` | `mean`, `sigma`, optional `min`/`max` | Log-normal. |
| `choice` | `values`, optional `weights` | Discrete weighted choice. |

The sampled visibility *V* is converted to the attenuation coefficient: **k = -ln(C_t) / V**.

### Heterogeneous Noise Fields

Both `k_hetero` and `ls_hetero` use Perlin FBM (fractional Brownian motion) to generate spatially-varying factor fields:

```json
"k_hetero": {
  "scales": "auto",
  "min_scale": 2,
  "max_scale": null,
  "min_factor": 0.0,
  "max_factor": 1.0,
  "normalize_to_mean": true
}
```

The noise field (values in [0, 1]) is mapped to a factor field: `factor(x) = min_factor + (max_factor - min_factor) * noise(x)`. When `normalize_to_mean` is `true`, the factor field is rescaled so its spatial mean equals 1.0, preserving the overall fog density while introducing spatial variation.

| Parameter | Effect |
|---|---|
| `min_factor` / `max_factor` | Range of the multiplicative factor. |
| `normalize_to_mean` | Rescale factors so the image-wide mean equals the base value. Recommended for `k_hetero`. |
| `scales` / `min_scale` / `max_scale` | Control spatial frequency content. |

### Fog Output

Foggy images are saved as PNG files organised by model name:

```
<output_path>/
  uniform/
    beta_0.0374_airlight_0.353_0.784_1_rgb_00000.png
    config.json
  heterogeneous_k/
    ...
```

---

## Sky-Depth Transform

Overrides depth values in sky regions with a configurable constant. Useful for datasets where sky depth is encoded as zero or infinity and needs to be normalised to a large finite value.

### Sky-Depth Config

```json
{
  "sky_depth_value": 1000.0
}
```

| Field | Description |
|---|---|
| `sky_depth_value` | Depth value assigned to all sky pixels. Defaults to `1000.0`. |

### Sky-Depth Output

Depth maps are saved as `.npy` float32 files preserving the original directory hierarchy.

---

## Radial Transform

Converts planar (z-buffer) depth to radial (Euclidean) depth using camera intrinsics. For each pixel *(u, v)*:

```
d_radial(u, v) = d_planar(u, v) * sqrt(((u - cx)/fx)^2 + ((v - cy)/fy)^2 + 1)
```

### Radial Config

```json
{}
```

No special parameters are required. The transform reads intrinsics from the `intrinsics` hierarchical modality.

### Radial Output

Depth maps are saved as `.npy` float32 files preserving the original directory hierarchy.
