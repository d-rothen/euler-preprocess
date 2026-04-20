from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from euler_preprocess.common.io import load_json
from euler_preprocess.common.logging import get_logger, progress_bar
from euler_preprocess.common.normalize import (
    _to_numpy,
    normalize_depth,
    normalize_sky_mask,
)
from euler_preprocess.common.output import LegacyOutputBackend
from euler_preprocess.common.transform import Transform


class SkyDepthTransform(Transform):
    """Override depth values in sky regions with a constant.

    Config JSON keys:
        ``sky_depth_value`` (float): Depth value to assign to sky pixels.
            Defaults to ``1000.0``.
        ``sanity_check_samples`` (int): Number of leading samples for which a
            detailed diagnostic breakdown of depth + sky mask is logged.
            Defaults to ``3``.  Set to ``0`` to disable.

    Each output sample is saved as a ``.npy`` float32 depth map.
    """

    REQUIRED_MODALITIES: ClassVar[set[str]] = {"depth", "semantic_segmentation"}
    SOURCE_MODALITY: ClassVar[str] = "depth"
    OUTPUT_SLOT: ClassVar[str] = "depth"

    DEFAULT_SANITY_CHECK_SAMPLES: ClassVar[int] = 3

    def __init__(
        self,
        config_path: str,
        out_path: str,
        output_backend: Any | None = None,
        strict: bool = False,
    ) -> None:
        self.config_path = Path(config_path)
        self.output_backend = output_backend or LegacyOutputBackend(out_path)
        self.out_path = self.output_backend.root

        self.config = load_json(self.config_path)
        self.sky_depth_value = float(self.config.get("sky_depth_value", 1000.0))
        self.sanity_check_samples = max(
            0,
            int(
                self.config.get(
                    "sanity_check_samples", self.DEFAULT_SANITY_CHECK_SAMPLES
                )
            ),
        )
        self.strict = bool(strict)
        self.logger = get_logger()
        self.logger.info(
            "SkyDepthTransform: sky_depth_value=%s strict=%s output=%s",
            self.sky_depth_value,
            self.strict,
            self.out_path,
        )

    def run(self, samples: Iterable[dict]) -> list[Path]:
        try:
            total = len(samples)  # type: ignore[arg-type]
        except TypeError:
            samples = list(samples)
            total = len(samples)
        saved_paths: list[Path] = []
        bad_sanity_ids: list[str] = []
        sanity_window = min(self.sanity_check_samples, total)

        with progress_bar(total, "sky-depth", self.logger) as bar:
            for index, sample in enumerate(samples):
                if index < sanity_window:
                    if self._log_sanity_check(sample, index):
                        bad_sanity_ids.append(str(sample.get("id")))
                    if index + 1 == sanity_window:
                        self._finalize_sanity_window(bad_sanity_ids, sanity_window)

                depth = normalize_depth(sample["depth"])
                sky_mask = normalize_sky_mask(sample["semantic_segmentation"])
                depth[sky_mask] = self.sky_depth_value

                output_path = self._build_output_path(
                    sample["id"], full_id=sample.get("full_id"),
                )
                saved_paths.append(
                    self.output_backend.write(
                        sample,
                        depth,
                        default_path=output_path,
                    )
                )

                if bar is not None:
                    bar.update(1)

        self.output_backend.finalize()
        return saved_paths

    def _finalize_sanity_window(
        self, bad_sanity_ids: list[str], window: int
    ) -> None:
        """Summarise the sanity window and, in strict mode, abort on failures."""
        if window == 0:
            return
        if not bad_sanity_ids:
            self.logger.info(
                "sanity check: %d/%d samples look OK.", window, window
            )
            return

        self.logger.warning(
            "sanity check: %d/%d samples flagged as suspicious (ids: %s).",
            len(bad_sanity_ids),
            window,
            bad_sanity_ids,
        )
        if self.strict:
            raise RuntimeError(
                f"sky-depth --strict: {len(bad_sanity_ids)}/{window} "
                f"sanity-check samples failed (ids: {bad_sanity_ids}). "
                f"Aborting before processing the rest of the dataset. "
                f"See the sanity-check log entries above for details."
            )

    def _log_sanity_check(self, sample: dict, index: int) -> bool:
        """Log a detailed breakdown of a single sample to catch misconfiguration.

        Intended to run for the first few samples only so the normal processing
        loop is not slowed down.  The goal is to make it obvious at a glance
        whether ``semantic_segmentation`` is really a boolean sky mask (vs. a
        full multi-class class map, a one-hot stack, or an RGB class-colour
        image) and whether any sky pixels are being found at all.

        Returns ``True`` if the sample looks suspicious (0 sky pixels,
        >90%% sky pixels, non-2D normalized mask, or a normalization error),
        ``False`` otherwise.  The return value drives ``--strict`` mode.
        """
        log = self.logger
        sample_id = sample.get("id")
        full_id = sample.get("full_id")
        is_bad = False

        log.info("=" * 72)
        log.info(
            "sky-depth sanity check [sample %d/%d, id=%s, full_id=%s]",
            index + 1,
            self.sanity_check_samples,
            sample_id,
            full_id,
        )
        log.info("  sample keys: %s", sorted(sample.keys()))

        # --- raw depth ---
        try:
            raw_depth = _to_numpy(sample["depth"])
            log.info(
                "  depth (raw):              shape=%s dtype=%s",
                raw_depth.shape,
                raw_depth.dtype,
            )
            finite = raw_depth[np.isfinite(raw_depth)]
            if finite.size:
                log.info(
                    "    finite stats:           min=%.4g max=%.4g "
                    "mean=%.4g  (non-finite pixels: %d)",
                    float(finite.min()),
                    float(finite.max()),
                    float(finite.mean()),
                    int(raw_depth.size - finite.size),
                )
            else:
                log.warning(
                    "    depth has no finite values! (all NaN/inf)"
                )
        except Exception as exc:  # noqa: BLE001
            log.error("  depth inspection failed: %s", exc)

        # --- raw semantic_segmentation ---
        raw_seg = sample["semantic_segmentation"]
        try:
            seg_arr = _to_numpy(raw_seg)
        except Exception as exc:  # noqa: BLE001
            log.error(
                "  semantic_segmentation: failed to convert to numpy (%s). "
                "type=%s",
                exc,
                type(raw_seg).__name__,
            )
            return True

        log.info(
            "  semantic_segmentation (raw): shape=%s dtype=%s (python type=%s)",
            seg_arr.shape,
            seg_arr.dtype,
            type(raw_seg).__name__,
        )

        self._describe_segmentation_classes(seg_arr)

        # --- normalized sky mask ---
        try:
            sky_mask = normalize_sky_mask(raw_seg)
        except Exception as exc:  # noqa: BLE001
            log.error(
                "  normalize_sky_mask() failed: %s. "
                "The semantic_segmentation modality is probably not in the "
                "expected shape (H, W) / (1, H, W) / (C, H, W) boolean mask.",
                exc,
            )
            return True

        n_sky = int(sky_mask.sum())
        total = int(sky_mask.size)
        pct = (100.0 * n_sky / total) if total else 0.0
        log.info(
            "  normalized sky mask:         shape=%s dtype=%s "
            "sky_pixels=%d/%d (%.2f%%)",
            sky_mask.shape,
            sky_mask.dtype,
            n_sky,
            total,
            pct,
        )

        # --- verdict ---
        if sky_mask.ndim != 2:
            log.warning(
                "  SANITY CHECK: normalized sky mask is not 2-D (got ndim=%d). "
                "depth[sky_mask]=... will likely broadcast incorrectly or error. "
                "Expected a boolean (H, W) mask.",
                sky_mask.ndim,
            )
            is_bad = True
        if n_sky == 0:
            log.warning(
                "  SANITY CHECK: 0 sky pixels found in sample '%s'. "
                "If this persists across every sanity-check sample, the "
                "`semantic_segmentation` modality is most likely pointing to "
                "the wrong data, or needs preprocessing into a boolean sky "
                "mask before reaching this transform (euler-loading typically "
                "produces a boolean (H, W) mask when the modality's "
                "class-meta tags the sky class). Expected: True where sky is "
                "present, False elsewhere.",
                sample_id,
            )
            is_bad = True
        elif pct > 90.0:
            log.warning(
                "  SANITY CHECK: %.2f%% of pixels are marked as sky in sample "
                "'%s'. That is suspiciously high — check that the mask is not "
                "inverted or that `astype(bool)` is not accidentally treating "
                "every non-zero class id as sky.",
                pct,
                sample_id,
            )
            is_bad = True

        log.info("=" * 72)
        return is_bad

    def _describe_segmentation_classes(self, seg_arr: np.ndarray) -> None:
        """Log a compact class-distribution summary for the raw segmentation.

        Helps identify whether the array looks like a boolean mask,
        a per-pixel class index map, a one-hot stack, or an RGB class-colour
        image.  Keeps output bounded (top-N classes only).
        """
        log = self.logger
        MAX_CLASSES = 12
        try:
            if seg_arr.ndim == 3 and seg_arr.shape[-1] in (3, 4):
                # Looks like an RGB(A) class-colour image — summarise per-channel.
                log.info(
                    "    appears to be a %d-channel image (likely RGB "
                    "class-colour coding, not a boolean mask)",
                    seg_arr.shape[-1],
                )
                for c in range(seg_arr.shape[-1]):
                    ch = seg_arr[..., c]
                    log.info(
                        "    channel %d: min=%s max=%s unique=%d",
                        c,
                        ch.min(),
                        ch.max(),
                        min(int(np.unique(ch).size), 999),
                    )
                return

            if seg_arr.ndim == 3 and seg_arr.shape[0] not in (1,):
                # Possibly a CHW stack (one-hot or per-class logits).
                log.info(
                    "    appears to be a multi-channel stack (C=%d, H=%d, W=%d) "
                    "— possibly one-hot or per-class score maps. If sky is a "
                    "specific channel, it must be reduced to a (H, W) boolean "
                    "mask before this transform.",
                    seg_arr.shape[0],
                    seg_arr.shape[1],
                    seg_arr.shape[2],
                )
                return

            flat = seg_arr.reshape(-1)
            unique, counts = np.unique(flat, return_counts=True)
            total = int(flat.size)
            if unique.size == 1:
                log.info(
                    "    single-valued array: value=%s count=%d  (this will "
                    "produce either an all-True or all-False mask)",
                    unique[0],
                    counts[0],
                )
            elif unique.size == 2 and set(int(u) for u in unique if u in (0, 1)) == {0, 1}:
                log.info(
                    "    looks like a binary mask (values={0, 1}): "
                    "1-count=%d (%.2f%%)",
                    int(counts[unique.tolist().index(1)]),
                    100.0 * int(counts[unique.tolist().index(1)]) / total,
                )
            else:
                order = np.argsort(-counts)
                log.info(
                    "    class-id histogram (top %d of %d unique values):",
                    min(MAX_CLASSES, unique.size),
                    unique.size,
                )
                for i in order[:MAX_CLASSES]:
                    val = unique[i]
                    cnt = int(counts[i])
                    log.info(
                        "      class=%s  pixels=%d  (%.2f%%)",
                        val,
                        cnt,
                        100.0 * cnt / total,
                    )
                if unique.size > MAX_CLASSES:
                    log.info(
                        "      ... (%d more classes omitted)",
                        unique.size - MAX_CLASSES,
                    )
                log.info(
                    "    NOTE: if this looks like a multi-class class-id map "
                    "rather than a boolean mask, the sky class must be "
                    "isolated (e.g. via euler-loading class-meta) before "
                    "reaching this transform — otherwise every non-zero "
                    "class id will be treated as 'sky' by astype(bool)."
                )
        except Exception as exc:  # noqa: BLE001
            log.error("    class-distribution summary failed: %s", exc)

    def _build_output_path(
        self, sample_id: str, full_id: str | None = None,
    ) -> Path:
        base = self.out_path
        if full_id:
            parts = [p for p in full_id.split("/") if p]
            if len(parts) > 1:
                base = base.joinpath(*parts[:-1])
        return base / f"{sample_id}.npy"
