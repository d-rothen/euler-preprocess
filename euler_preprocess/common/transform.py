from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar


class Transform(ABC):
    """Base class for all preprocessing transforms.

    Subclasses declare the modalities they need via class variables and
    implement :meth:`run` to process samples.
    """

    REQUIRED_MODALITIES: ClassVar[set[str]] = set()
    REQUIRED_HIERARCHICAL_MODALITIES: ClassVar[set[str]] = set()
    SOURCE_MODALITY: ClassVar[str | None] = None
    OUTPUT_SLOT: ClassVar[str | None] = None
    OUTPUT_INDEX_META_OVERRIDES: ClassVar[dict[str, object]] = {}

    @abstractmethod
    def __init__(self, config_path: str, out_path: str) -> None: ...

    @abstractmethod
    def run(self, samples: Iterable[dict]) -> list[Path]: ...
