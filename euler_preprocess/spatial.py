"""Opt-in deterministic spatial producer using loading's shared execution path."""

from __future__ import annotations

from copy import deepcopy

from euler_loading import SamplePreprocessor, execution_profile

from .common.dataset import build_dataset
from .common.output import prepare_output_backends


def export_spatial(config):
    """Generate explicitly selected fields/variants, with content-verified receipts.

    ``samples`` is an optional list of source dataset indices. ``mapping`` is an
    optional {variant: {source_full_id: output_full_id}}; otherwise one variant
    preserves source IDs and several variants append their explicit variant IDs.
    """
    config = deepcopy(config)
    variants = config.get("spatial_variants") or {
        "default": {
            "preprocessing": config["preprocessing"],
            "bindings": config["bindings"],
        }
    }
    dataset_config = {
        key: value for key, value in config.items() if key != "preprocessing"
    }
    # Bound fields with a spatial layout are per-sample images; the role of the
    # others (e.g. intrinsics) is inferred from their ds-crawler indexes.
    per_sample = {
        name
        for options in variants.values()
        for name, binding in options["bindings"]["fields"].items()
        if "H" in binding["profile"]["layout"]
    }
    dataset = build_dataset(dataset_config, per_sample)
    plans = {}
    for variant, options in variants.items():
        bindings = deepcopy(options["bindings"])
        sources = dataset.describe_transform_sources(
            bindings["fields"], revisions=bindings["revisions"]
        )
        plans[variant] = SamplePreprocessor.from_config(
            options["preprocessing"]
        ).export_descriptor(
            fields=bindings["fields"],
            image_planes=bindings["image_planes"],
            sources=sources,
            execution=execution_profile(bindings.get("backend", "torch_cpu")),
        )
    dataset.enable_transform_capture(plans)
    indices = config.get("samples", list(range(len(dataset))))
    if (
        not isinstance(indices, list)
        or not indices
        or any(type(i) is not int or i < 0 or i >= len(dataset) for i in indices)
    ):
        raise ValueError(
            "spatial samples must be a nonempty list of valid dataset indices"
        )

    def output_id(sample, variant):
        mapping = config.get("mapping", {}).get(variant)
        if mapping is not None:
            return mapping[sample["full_id"]]
        return (
            sample["full_id"]
            if len(variants) == 1
            else sample["full_id"] + "/" + variant
        )

    # Resolve expected IDs without decoding or keeping an entire dataset in RAM.
    expected = []
    for variant, plan in plans.items():
        reference = plan["recipe"]["reference_field"]
        for i in indices:
            fid = dataset.sample_full_id(i, reference)
            expected.append(output_id({"full_id": fid}, variant))
    if len(set(expected)) != len(expected):
        raise ValueError(
            "spatial output mappings must be unique for all selected variants"
        )
    backends = {}
    try:
        materialization_outputs = {
            field: {**options, "derivation": plans, "expected_full_ids": expected}
            for field, options in config["outputs"].items()
        }
        backends = prepare_output_backends(
            {"materialization_outputs": materialization_outputs}, dataset, object
        )
        for variant in plans:
            for i in indices:
                sample = dataset.capture_sample(i, variant_id=variant)
                for field, backend in backends.items():
                    backend.write(
                        sample,
                        sample[field],
                        output_full_id=output_id(sample, variant),
                        variant_id=variant,
                        sidecar=config.get("receipt_sidecars", False),
                    )
        for backend in backends.values():
            backend.finalize()
    finally:
        for backend in backends.values():
            backend.dataset_writer.close()
    return {field: str(backend.root) for field, backend in backends.items()}
