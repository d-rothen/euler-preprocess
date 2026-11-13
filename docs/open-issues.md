# Open issues

Known gaps still to fix. Each entry says what is wrong, where, how much it
matters, and the intended fix.

- [Geometric capture effects warp the RGB only](#geometric-capture-effects-warp-the-rgb-only)

## Geometric capture effects warp the RGB only

*Recorded 2026-11-13, from the UrbanSyn fog run.*

### What is wrong

The camera-simulation stage (`capture`, `euler_preprocess/fog/capture.py`)
applies several effects that **move pixels**. It applies them to the rendered
RGB alone. Every other modality of the sample keeps the source's pixel grid:
- depth (planar or radial);
- sparse depth and point clouds;
- semantic segmentation and sky masks;
- normals;
- the intrinsics.

Training and evaluation join those modalities with the fogged RGB by sample ID,
so the image and its labels disagree wherever the RGB was moved.

| Effect | Config key (`optics`) | Code | Pixel motion |
|---|---|---|---|
| Radial lens distortion | `lens_distortion` (k1), `lens_distortion_k2` | sampled at `capture.py` ~l.208 / ~l.288; `_lens_distort_np` / `_lens_distort_torch` (~l.4001 / ~l.4019) | Global radial warp about (cx, cy). With k1 in [−0.012, 0.006] on UrbanSyn (2048×1024, fx 1730): median 2.4 px at the corners, up to 6 px; 80 % of frames > 1 px |
| Lens / windshield droplets | `droplets` (`refraction_px`, `probability`) | `_apply_droplets_np` (~l.4569) and its torch twin | Local refraction of 0.8–3.2 px inside each droplet |
| Chromatic aberration | `chromatic_aberration_px` | `_chromatic_aberration_np` / `_torch` (~l.4038 / ~l.4059) | Red and blue shifted in opposite directions by up to ~0.3 px; green untouched |
| Depth-dependent fringing | `depth_chromatic_fringing` | `_apply_depth_chromatic_fringing_np` / `_torch` (called at ~l.224 / ~l.304) | Red/blue shifts up to ~0.55 px |
| Motion blur | `motion_blur` | `_motion_blur_np` / `_torch` (~l.4249 / ~l.4266) | Centred kernel of 2–4 px: smears, no net shift |

Photometric effects (ISP, tone mapping, noise, vignetting, bloom, glare,
windshield haze, JPEG) do not move pixels and are not affected.

**What happened.** The first UrbanSyn fog run with the user's
`dense_gloomy_daylight_dashcam` profile used lens distortion in −0.012…0.006
and droplets in up to 12 % of foggy frames. Its RGB was 2–6 px off its depth
and segmentation at the image borders. The run was stopped, and the config was
changed:
- `lens_distortion` set to uniform 0..0;
- droplet `probability` set to 0 in the scenario overrides;
- the backup is `/cluster/work/igp_psr/drothenpiele/configs/preprocess-fog-config.json.bak-20261113`.

That avoids the problem rather than fixing it.

### Intended fix: every geometric effect moves every spatial modality

A pixel-moving effect is a mapping from source to output pixel positions. Apply
that mapping to every modality that lives on the image grid, or refuse to run.

1. **Represent each geometric effect as an explicit mapping.** Model lens
   distortion, droplet refraction and chromatic shifts as a sampling map
   (inverse map or displacement field) built once per sample. Apply it the same
   way to the RGB and to every other spatial modality written or joined with
   it.
2. **Resample each modality kind correctly.** This should go through
   euler-loading's existing field-kind machinery (`SamplePreprocessor`, the
   invalid-aware resize):
   - **RGB, intensity:** bilinear.
   - **Depth and distance:** invalid-aware nearest or bilinear, never mixing
     across invalid pixels or depth edges.
   - **Sparse depth and point clouds:** move each measurement to its new pixel
     (forward-map the projected points), not interpolate.
   - **Segmentation, sky mask, class IDs:** nearest.
   - **Normals:** warp, then re-express in the new camera frame if the effect
     changes the ray directions (it does for lens distortion).
3. **Keep the camera description truthful.** A distorted image is no longer
   described by the pinhole K alone. Choose one:
   - **(a)** Keep K and **record the distortion model and coefficients** in
     the output (head meta and per-file attributes, e.g.
     `{"model": "radial_k1k2", "k1": …, "k2": …, "center": [cx, cy]}`), so
     consumers can undistort or project correctly. euler-loading's `intrinsics`
     head meta already has `distortion: null` for this.
   - **(b)** Warp all modalities and keep a pinhole K, which is only exact for
     effects that do not change the projection model.

   Lens distortion needs (a). Droplets and chromatic shifts are local or
   per-channel; they need the warped modalities, and are otherwise only
   recorded.
4. **Fail loudly in the meantime.**
   - If a config enables any pixel-moving effect while other spatial
     modalities are in the dataset config, stop with an error listing the
     effects and the affected modalities.
   - Add an explicit opt-out (e.g. `capture.allow_rgb_only_geometry: true`)
     for users who want RGB-only augmentation and know the labels will not
     follow.
5. **Record the sampled parameters per file.** In sample mode the fog writer
   stores no per-file attributes beyond the source's: no scenario, visibility
   or β, airlight, k1, droplets or JPEG quality. The UrbanSyn run had to
   reconstruct them from the seed afterwards (`gpu_batching` draws one set per
   batch of 8). Write them to each output's attributes, as `augmentations`
   mode already does with `fog_augmentation`.
6. **Tests.**
   - A synthetic checkerboard RGB with a matching depth and label grid, run
     through each geometric effect: the warped labels must stay aligned with
     the warped RGB to ≤0.5 px.
   - Sparse points must land on the warped features.
   - With all effects at zero, the output must be bit-identical to the input
     geometry.

### Related

- euler-loading `docs/open-issues.md`: the planar ↔ radial depth conversion
  drops the intrinsics' skew. Its shared-helper fix belongs to the same
  "camera model is honoured everywhere" theme.
- The fog output's dataset head is copied from the source: same dataset
  id/name, `gt: true`, and the source's rgb meta, e.g. Hypersim's "stored:
  linear HDR colour". Each file also inherits source attributes such as
  `tonemap_scale`. Derived datasets should describe what they hold and mark
  their provenance.
