from __future__ import annotations

import logging

import pytest

from euler_preprocess.cli import _select_configured_samples


class _Dataset:
    def __init__(self, size: int = 3) -> None:
        self.items = [
            {"id": f"frame_{index:03d}", "full_id": f"/Scene/Camera/frame_{index:03d}"}
            for index in range(size)
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        return self.items[index]


def test_sample_selection_returns_only_configured_index() -> None:
    dataset = _Dataset()

    selected = _select_configured_samples(
        {"sample": 1},
        dataset,
        logging.getLogger("test"),
    )

    assert selected == [dataset[1]]


def test_sample_selection_without_config_returns_dataset() -> None:
    dataset = _Dataset()

    selected = _select_configured_samples({}, dataset, logging.getLogger("test"))

    assert selected is dataset


def test_samples_selection_accepts_explicit_indices() -> None:
    dataset = _Dataset(size=5)

    selected = _select_configured_samples(
        {"samples": [0, 3]},
        dataset,
        logging.getLogger("test"),
    )

    assert len(selected) == 2
    assert list(selected) == [dataset[0], dataset[3]]


def test_samples_selection_accepts_sparse_slice() -> None:
    dataset = _Dataset(size=6)

    selected = _select_configured_samples(
        {"samples": {"step": 2}},
        dataset,
        logging.getLogger("test"),
    )

    assert len(selected) == 3
    assert list(selected) == [dataset[0], dataset[2], dataset[4]]


def test_samples_selection_accepts_start_stop_step_and_count() -> None:
    dataset = _Dataset(size=10)

    selected = _select_configured_samples(
        {"samples": {"start": 1, "stop": 9, "step": 2, "count": 3}},
        dataset,
        logging.getLogger("test"),
    )

    assert list(selected) == [dataset[1], dataset[3], dataset[5]]


@pytest.mark.parametrize("value", [-1, True, 1.5, "1"])
def test_sample_selection_rejects_invalid_values(value) -> None:
    with pytest.raises(ValueError, match="sample must be a non-negative integer"):
        _select_configured_samples(
            {"sample": value},
            _Dataset(),
            logging.getLogger("test"),
        )


def test_sample_selection_rejects_combined_sample_and_samples() -> None:
    with pytest.raises(ValueError, match="either sample or samples"):
        _select_configured_samples(
            {"sample": 1, "samples": {"step": 2}},
            _Dataset(),
            logging.getLogger("test"),
        )


def test_sample_selection_rejects_out_of_range_index() -> None:
    with pytest.raises(IndexError, match="out of range"):
        _select_configured_samples(
            {"sample": 3},
            _Dataset(size=3),
            logging.getLogger("test"),
        )


@pytest.mark.parametrize(
    "selection",
    [
        [],
        {"step": 0},
        {"start": -1},
        {"count": 0},
        {"unknown": 1},
        "every_2",
    ],
)
def test_samples_selection_rejects_invalid_specs(selection) -> None:
    with pytest.raises((ValueError, IndexError)):
        _select_configured_samples(
            {"samples": selection},
            _Dataset(size=3),
            logging.getLogger("test"),
        )
