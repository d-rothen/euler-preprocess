from __future__ import annotations

from collections import Counter

from euler_preprocess.common.logging import get_logger


def _make_modality(entry: str | dict, function: str | None = None):
    """Build a ``Modality``; an entry's own ``function`` replaces *function*."""
    from euler_loading import Modality

    if isinstance(entry, str):
        return Modality(entry, function=function)
    options = {
        key: entry[key]
        for key in (
            "split",
            "metadata_scope",
            "collapse_single",
            "cache",
            "applies_to",
            "hierarchy_scope",
            "function",
        )
        if key in entry
    }
    options.setdefault("function", function)
    return Modality(entry["path"], **options)


def _load_index_tree(name: str, modality) -> dict:
    """Load a modality's ds-crawler tree the same way ``MultiModalDataset`` does."""
    from ds_crawler import index_dataset_from_path
    from euler_loading._ds_crawler_utils import load_index_output
    from euler_loading.dataset import _get_index_tree, _resolve_metadata_scope

    modality = _resolve_metadata_scope(name, modality)
    index = load_index_output(
        modality.path,
        split=modality.split,
        metadata_scope=modality.metadata_scope,
        index_dataset_from_path_fn=index_dataset_from_path,
    )
    return _get_index_tree(index)


def _check_attachment(
    name: str, seed: str, tree: dict, samples_per_level: Counter
) -> None:
    """Check that hierarchical modality *name* applies to the samples by tree position."""
    from euler_loading.indexing import collect_hierarchical_files, match_hierarchical_files

    files_by_level = collect_hierarchical_files(tree)
    attached = {
        level: len(match_hierarchical_files(level, files_by_level))
        for level in samples_per_level
    }
    covered = sum(count for level, count in samples_per_level.items() if attached[level])
    if not covered:
        raise ValueError(
            f"Modality '{name}' shares no sample IDs with '{seed}' and none of its "
            "files sit at or above the hierarchy level of any sample, so it can be "
            "neither joined per sample nor matched hierarchically. Check that its "
            f"ds-crawler IDs and hierarchy follow the same conventions as '{seed}'."
        )
    logger = get_logger()
    total = sum(samples_per_level.values())
    log = logger.info if covered == total else logger.warning
    log(
        "Modality '%s' is hierarchical (no sample IDs shared with '%s'); it applies "
        "to %d/%d samples by hierarchy position.",
        name,
        seed,
        covered,
        total,
    )
    most = max(attached.values())
    if most > 1:
        logger.warning(
            "Hierarchical modality '%s' attaches up to %d files to one sample. If it "
            "holds per-sample data, its ds-crawler IDs do not match '%s'.",
            name,
            most,
            seed,
        )


def _resolve_roles(
    modalities: dict, per_sample: set[str]
) -> tuple[list[str], list[str]]:
    """Split modality names into regular and hierarchical ones.

    The role follows from the ds-crawler indexes.  A modality is regular
    (joined per sample) when its qualified file IDs -- hierarchy path plus
    file ID -- overlap those of the seed modality.  Otherwise it is
    hierarchical (matched by tree position, like per-scene intrinsics) and
    must sit at or above the hierarchy level of at least one sample.

    *per_sample* modalities are regular by contract and the first one is the
    seed.  Without one, the seed is the modality aligned with the most others,
    then the one with the most files, then the first listed.
    """
    from euler_loading.indexing import collect_files

    names = list(modalities)
    if len(names) < 2 or per_sample.issuperset(names):
        return names, []

    trees = {name: _load_index_tree(name, modalities[name]) for name in names}
    # Qualified sample ID -> hierarchy path, keyed like euler-loading's lookups.
    sample_ids = {}
    for name, tree in trees.items():
        sample_ids[name] = {
            "/".join(r.hierarchy_path + (r.file_entry["id"],)): r.hierarchy_path
            for r in collect_files(tree)
        }

    def aligned(name: str, other: str) -> bool:
        return not sample_ids[name].keys().isdisjoint(sample_ids[other].keys())

    anchors = [name for name in names if name in per_sample]
    seed = anchors[0] if anchors else max(
        names,
        key=lambda name: (
            sum(aligned(other, name) for other in names),
            len(sample_ids[name]),
        ),
    )
    regular = [
        name
        for name in names
        if name in per_sample or name == seed or aligned(name, seed)
    ]
    hierarchical = [name for name in names if name not in regular]
    get_logger().info(
        "Modality roles inferred from ds-crawler indexes: per-sample %s; "
        "hierarchical %s",
        ", ".join(regular),
        ", ".join(hierarchical) or "none",
    )

    # Without common samples MultiModalDataset raises its own, clearer error.
    common = set(sample_ids[seed]).intersection(*(sample_ids[name] for name in regular))
    if common:
        samples_per_level = Counter(sample_ids[seed][sample_id] for sample_id in common)
        for name in hierarchical:
            _check_attachment(name, seed, trees[name], samples_per_level)
    return regular, hierarchical


def build_dataset(
    config: dict,
    required_modalities: set[str],
    required_hierarchical: set[str] | None = None,
    modality_functions: dict[str, str] | None = None,
):
    """Build a ``MultiModalDataset`` from a config dict.

    Each modality entry is either a plain path string or a dict with
    ``path`` and optional ``split``.  Loader resolution (which function to
    call, which module to use) is handled by euler-loading via the
    ds-crawler index at each path — point the config at a path whose
    index declares the function you want (e.g. a ``sky_mask`` index for
    boolean sky masks vs. a ``class_segmentation`` index for raw class
    id maps).

    Every modality can be listed under ``modalities``: whether it is joined
    per sample or matched hierarchically (e.g. per-scene intrinsics) is
    inferred from the ds-crawler indexes.  Entries under the legacy
    ``hierarchical_modalities`` key are pooled with them and resolved the
    same way.  *required_modalities* are per-sample inputs and anchor that
    inference; *required_hierarchical* names further required modalities,
    whose role is inferred like any other.

    *modality_functions* maps a modality to the loader function to use from
    the module its dataset head names (a transform's ``MODALITY_FUNCTIONS``),
    e.g. ``sky_mask`` for a ``class_segmentation`` index; the sky then comes
    from the head's ``skyclass``.  A modality entry's own ``function`` key
    overrides it, and ``"function": null`` keeps the head's function.
    """
    from euler_loading import MultiModalDataset, SamplePreprocessor

    raw_modalities = config.get("modalities", {})
    raw_hierarchical = config.get("hierarchical_modalities", {})

    duplicated = raw_modalities.keys() & raw_hierarchical.keys()
    if duplicated:
        raise ValueError(
            "Modalities listed under both 'modalities' and "
            f"'hierarchical_modalities': {', '.join(sorted(duplicated))}"
        )
    configured = {**raw_modalities, **raw_hierarchical}

    required = set(required_modalities) | set(required_hierarchical or ())
    missing = required - configured.keys()
    if missing:
        raise ValueError(
            f"Missing required modalities in config: {', '.join(sorted(missing))}. "
            f"'modalities' must contain at least: {', '.join(sorted(required))}"
        )

    functions = modality_functions or {}
    modalities = {
        name: _make_modality(entry, functions.get(name))
        for name, entry in configured.items()
    }
    regular, hierarchical = _resolve_roles(modalities, set(required_modalities))
    logger = get_logger()
    for name in regular:
        if name in raw_hierarchical:
            logger.warning(
                "Modality '%s' is listed under 'hierarchical_modalities' but holds "
                "per-sample data; loading it per sample.",
                name,
            )

    return MultiModalDataset(
        modalities={name: modalities[name] for name in regular},
        hierarchical_modalities={name: modalities[name] for name in hierarchical}
        or None,
        transforms=[SamplePreprocessor.from_config(config["preprocessing"])]
        if config.get("preprocessing")
        else None,
    )
