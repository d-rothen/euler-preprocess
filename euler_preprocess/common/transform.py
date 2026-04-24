from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any, ClassVar


class Transform(ABC):
    """Base class for all preprocessing transforms.

    Subclasses declare the modalities they need via class variables and
    implement :meth:`run` to process samples.

    Output slots:
        Most transforms produce a single output (the *primary* slot, declared
        via :attr:`OUTPUT_SLOT`).  Transforms that produce additional auxiliary
        outputs (e.g. fog β / L_s maps) declare them in :attr:`OUTPUT_SLOTS`
        together with per-slot specs in :attr:`OUTPUT_SLOT_SPECS`.  Auxiliary
        slots are opt-in: they are only written when the dataset config's
        ``pipeline.output_targets`` includes a matching entry.
    """

    REQUIRED_MODALITIES: ClassVar[set[str]] = set()
    REQUIRED_HIERARCHICAL_MODALITIES: ClassVar[set[str]] = set()
    SOURCE_MODALITY: ClassVar[str | None] = None
    OUTPUT_SLOT: ClassVar[str | None] = None
    OUTPUT_SLOTS: ClassVar[tuple[str, ...]] = ()
    OUTPUT_SLOT_SPECS: ClassVar[dict[str, Any]] = {}
    OUTPUT_INDEX_META_OVERRIDES: ClassVar[dict[str, object]] = {}

    @abstractmethod
    def __init__(self, config_path: str, out_path: str) -> None: ...

    @abstractmethod
    def run(self, samples: Iterable[dict]) -> list[Path]: ...
