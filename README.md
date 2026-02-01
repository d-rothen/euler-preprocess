# Fog Generation

Physically-based fog augmentation for multi-modal image datasets. Takes aligned RGB, depth, and semantic segmentation images and produces realistic foggy versions using the Koschmieder atmospheric scattering model with support for spatially-heterogeneous fog.

## Usage

```bash
python main.py --config configs/example_dataset_config.json
```

## Configuration

Two configuration files are required: a **dataset config** and a **fog config**.

### Dataset Config

Points to the input data and the fog parameters.

```json
{
  "fog_config_path": "configs/run1.json",
  "output_path": "/path/to/output",
  "modalities": {
    "rgb": "/path/to/rgb",
    "depth": "/path/to/depth",
    "classSegmentation": "/path/to/classSegmentation"
  },
  "sky_color": [90, 200, 255]
}
```

| Field | Description |
|---|---|
| `fog_config_path` | Path to the fog generation config (see below). |
| `output_path` | Directory where foggy images are written. |
| `modalities.rgb` | Root directory of RGB images. |
| `modalities.depth` | Root directory of depth maps (values in **metres**). |
| `modalities.classSegmentation` | Root directory of semantic segmentation maps. |
| `sky_color` | RGB triplet identifying sky pixels in the segmentation maps. Default `[90, 200, 255]` (VKITTI). |

### Fog Config

Controls the fog simulation itself.

```json
{
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
| `seed` | Random seed for reproducibility. `null` for non-deterministic. |
| `depth_scale` | Multiplier applied to depth values after loading. |
| `resize_depth` | Resize the depth map to match the RGB resolution (bilinear). |
| `contrast_threshold` | Threshold *C_t* used in the visibility-to-attenuation conversion (default `0.05`). |
| `device` | `"cpu"`, `"cuda"`, `"mps"`, or `"gpu"` (alias for cuda). |
| `gpu_batch_size` | Batch size when running on GPU. Uniform-model samples are batched; heterogeneous samples are processed individually. |

## Fog Model

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

## How Each Modality is Used

### RGB

The clean scene image. Normalised to float32 in [0, 1]. This is the *I(x)* term in the fog equation -- it gets blended with the airlight according to transmittance.

### Depth

A per-pixel depth map in **metres**. Provides the *d(x)* term in the transmittance calculation `t(x) = exp(-k * d(x))`. Pixels with greater depth receive more fog. Invalid values (NaN, inf, negative) are clamped to zero (treated as infinitely close, receiving no fog).

### Segmentation (Sky Mask)

The semantic segmentation map is used exclusively to derive a **sky mask**: a boolean per-pixel mask where the segmentation colour matches `sky_color`. The sky mask has two purposes:

1. **Airlight estimation** -- when `atmospheric_light` is `"from_sky"`, the mean RGB of all sky pixels in the clean image is used as the airlight colour *L_s*. This makes the fog colour match the actual sky appearance of each scene.
2. **Scene-consistent appearance** -- because sky pixels already represent the atmospheric light colour, using them as *L_s* ensures the fog blends naturally into the horizon.

## Model Selection

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

Four models are available, described below.

### `uniform`

Constant *k* and constant *L_s* across the entire image. The simplest and fastest model -- standard homogeneous fog.

### `heterogeneous_k`

Spatially-varying attenuation coefficient with constant airlight. Simulates patchy fog where density varies across the scene (e.g. fog banks, ground fog).

### `heterogeneous_ls`

Constant attenuation coefficient with spatially-varying airlight. Simulates variation in the scattered light colour across the image.

### `heterogeneous_k_ls`

Both *k* and *L_s* vary spatially. The most expressive model, combining density and colour variation.

## Visibility Distribution

Each model specifies a `visibility_m` distribution from which a visibility distance (in metres) is sampled per image. Supported distributions:

| `dist` | Parameters | Description |
|---|---|---|
| `constant` | `value` | Fixed value. |
| `uniform` | `min`, `max` | Uniform random in range. |
| `normal` | `mean`, `std`, optional `min`/`max` | Gaussian, optionally clamped. |
| `lognormal` | `mean`, `sigma`, optional `min`/`max` | Log-normal. |
| `choice` | `values`, optional `weights` | Discrete weighted choice. |

The sampled visibility *V* is converted to the attenuation coefficient: **k = -ln(C_t) / V**.

## Heterogeneous Attenuation (`k_hetero`)

The `k_hetero` block controls how the scalar *k* is modulated into a spatially-varying 2D field.

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

**Noise generation.** A fractional Brownian motion (FBM) field is synthesised by summing multiple octaves of improved Perlin noise. Each octave operates at a different spatial scale and is weighted by `log2(scale)^2`, giving larger scales (low-frequency variation) more influence.

**Scale resolution.** When `scales` is `"auto"`, octave scales are generated as powers of 2 from `min_scale` up to `max_scale` (defaults to `max(H, W)`). You can also supply an explicit list of integers.

**Modulation.** The noise field (values in [0, 1]) is mapped to a factor field:

```
factor(x) = min_factor + (max_factor - min_factor) * noise(x)
```

When `normalize_to_mean` is `true`, the factor field is rescaled so its spatial mean equals 1.0. This preserves the overall fog density (mean *k* stays at `k_mean`) while introducing spatial variation. The final field is:

```
k(x) = k_mean * factor(x)
```

| Parameter | Effect |
|---|---|
| `min_factor` / `max_factor` | Range of the multiplicative factor. `[0, 1]` means fog density ranges from clear to the base value. `[0.5, 1.5]` means +/-50% variation. |
| `normalize_to_mean` | When `true`, rescales factors so the image-wide mean *k* equals `k_mean`. Recommended for `k_hetero` to keep the average fog consistent with the sampled visibility. |
| `scales` / `min_scale` / `max_scale` | Control the spatial frequency content. Smaller scales produce fine-grained patchiness; larger scales produce broad fog banks. |

## Heterogeneous Airlight (`ls_hetero`)

The `ls_hetero` block modulates the base airlight colour into a spatially-varying (H, W, 3) field using the same Perlin FBM approach.

```json
"ls_hetero": {
  "scales": "auto",
  "min_scale": 2,
  "max_scale": null,
  "min_factor": 0.0,
  "max_factor": 1.0,
  "normalize_to_mean": false
}
```

The base airlight *L_s* (a 3-channel RGB value, either estimated from sky pixels or sampled from the config) is multiplied by the same type of factor field:

```
L_s(x) = L_s_base * factor(x)
```

The factor field is scalar (single-channel), so all three colour channels are scaled uniformly at each pixel -- this varies the **intensity** of the airlight across the image without shifting its hue.

| Parameter | Effect |
|---|---|
| `min_factor` / `max_factor` | Range of brightness modulation. `[0, 1]` means airlight ranges from black to the base colour. |
| `normalize_to_mean` | Typically `false` for `ls_hetero`. Setting it to `true` would preserve the mean airlight brightness but is less physically motivated than for *k*. |
| `scales` / `min_scale` / `max_scale` | Same spatial frequency control as `k_hetero`. |

## Output

Foggy images are saved as PNG files organised by model name:

```
<output_path>/
  uniform/
    beta_0.0374_airlight_0.353_0.784_1_rgb_00000.png
    config.json
  heterogeneous_k/
    ...
```

The filename encodes the mean attenuation coefficient (`beta`) and airlight RGB used for that image. Each model subdirectory also contains a `config.json` with the resolved parameters.
