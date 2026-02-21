from __future__ import annotations


def build_dataset(
    config: dict,
    required_modalities: set[str],
    required_hierarchical: set[str] | None = None,
):
    """Build a ``MultiModalDataset`` from a config dict.

    Args:
        config: Top-level dataset config containing ``modalities`` and
            optionally ``hierarchical_modalities`` mappings.
        required_modalities: Set of modality names that must be present.
        required_hierarchical: Optional set of hierarchical modality names
            that must be present.

    Returns:
        A ``MultiModalDataset`` instance.
    """
    from euler_loading import Modality, MultiModalDataset

    modality_paths = config.get("modalities", {})
    hierarchical_paths = config.get("hierarchical_modalities", {})

    missing = required_modalities - modality_paths.keys()
    if missing:
        raise ValueError(
            f"Missing required modalities in config: {', '.join(sorted(missing))}. "
            f"'modalities' must contain at least: {', '.join(sorted(required_modalities))}"
        )

    if required_hierarchical:
        missing_h = required_hierarchical - hierarchical_paths.keys()
        if missing_h:
            raise ValueError(
                f"Missing required hierarchical modalities in config: "
                f"{', '.join(sorted(missing_h))}. 'hierarchical_modalities' must "
                f"contain at least: {', '.join(sorted(required_hierarchical))}"
            )

    modalities = {
        name: Modality(path) for name, path in modality_paths.items()
    }
    hierarchical_modalities = {
        name: Modality(path) for name, path in hierarchical_paths.items()
    } or None

    return MultiModalDataset(
        modalities=modalities,
        hierarchical_modalities=hierarchical_modalities,
    )
