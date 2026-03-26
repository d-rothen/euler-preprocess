from __future__ import annotations


def _parse_modality_entry(entry: str | dict) -> dict:
    """Normalise a modality config entry to ``{path, split}``."""
    if isinstance(entry, str):
        return {"path": entry}
    return entry


def build_dataset(
    config: dict,
    required_modalities: set[str],
    required_hierarchical: set[str] | None = None,
):
    """Build a ``MultiModalDataset`` from a config dict.

    Args:
        config: Top-level dataset config containing ``modalities`` and
            optionally ``hierarchical_modalities`` mappings.  Each modality
            value may be a plain path string or a dict with ``path`` and
            an optional ``split`` key.
        required_modalities: Set of modality names that must be present.
        required_hierarchical: Optional set of hierarchical modality names
            that must be present.

    Returns:
        A ``MultiModalDataset`` instance.
    """
    from euler_loading import Modality, MultiModalDataset

    raw_modalities = config.get("modalities", {})
    raw_hierarchical = config.get("hierarchical_modalities", {})

    missing = required_modalities - raw_modalities.keys()
    if missing:
        raise ValueError(
            f"Missing required modalities in config: {', '.join(sorted(missing))}. "
            f"'modalities' must contain at least: {', '.join(sorted(required_modalities))}"
        )

    if required_hierarchical:
        missing_h = required_hierarchical - raw_hierarchical.keys()
        if missing_h:
            raise ValueError(
                f"Missing required hierarchical modalities in config: "
                f"{', '.join(sorted(missing_h))}. 'hierarchical_modalities' must "
                f"contain at least: {', '.join(sorted(required_hierarchical))}"
            )

    modalities = {}
    for name, entry in raw_modalities.items():
        parsed = _parse_modality_entry(entry)
        modalities[name] = Modality(parsed["path"], split=parsed.get("split"))

    hierarchical_modalities = {}
    for name, entry in raw_hierarchical.items():
        parsed = _parse_modality_entry(entry)
        hierarchical_modalities[name] = Modality(parsed["path"], split=parsed.get("split"))

    return MultiModalDataset(
        modalities=modalities,
        hierarchical_modalities=hierarchical_modalities or None,
    )
