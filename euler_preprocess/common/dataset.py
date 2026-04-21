from __future__ import annotations


def _make_modality(entry: str | dict):
    from euler_loading import Modality

    if isinstance(entry, str):
        return Modality(entry)
    return Modality(entry["path"], split=entry.get("split"))


def build_dataset(
    config: dict,
    required_modalities: set[str],
    required_hierarchical: set[str] | None = None,
):
    """Build a ``MultiModalDataset`` from a config dict.

    Each modality entry is either a plain path string or a dict with
    ``path`` and optional ``split``.  Loader resolution (which function to
    call, which module to use) is handled by euler-loading via the
    ds-crawler index at each path — point the config at a path whose
    index declares the function you want (e.g. a ``sky_mask`` index for
    boolean sky masks vs. a ``class_segmentation`` index for raw class
    id maps).
    """
    from euler_loading import MultiModalDataset

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

    modalities = {name: _make_modality(entry) for name, entry in raw_modalities.items()}
    hierarchical_modalities = {
        name: _make_modality(entry) for name, entry in raw_hierarchical.items()
    }

    return MultiModalDataset(
        modalities=modalities,
        hierarchical_modalities=hierarchical_modalities or None,
    )
