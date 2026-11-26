"""Fake installed packages for ``euler.*`` entry-point tests.

``install`` replaces euler-dataset-contract's entry-point discovery, the way
its own tests simulate installed distributions, and makes the transform
catalog load again from the fakes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from euler_dataset_contract import extensions

from euler_preprocess import registry


@dataclass
class FakeDist:
    name: str
    version: str

    @property
    def metadata(self) -> dict[str, str]:
        return {"Name": self.name}


@dataclass
class FakeEntryPoint:
    name: str
    value: str
    group: str = "euler.transforms"
    dist: Any = None


def plugin(name: str, value: str, distribution: str = "my-lab") -> FakeEntryPoint:
    return FakeEntryPoint(name, value, "euler.transforms", FakeDist(distribution, "0.1.0"))


def install(monkeypatch: pytest.MonkeyPatch, *entry_points: FakeEntryPoint) -> None:
    def raw(group: str) -> list[FakeEntryPoint]:
        return [ep for ep in entry_points if ep.group == group]

    monkeypatch.setattr(extensions, "_raw_entry_points", raw)
    monkeypatch.setattr(
        extensions,
        "_raw_entry_point_groups",
        lambda: sorted({ep.group for ep in entry_points}),
    )
    # Loaded lazily from the fakes; monkeypatch restores the real catalog after.
    monkeypatch.setattr(registry, "_catalog", None)
