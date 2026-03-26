from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def get_logger(name: str = "euler_preprocess") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def log_dataset_info(
    logger: logging.Logger,
    dataset_name: str,
    dataset_size: int,
    modality_info: dict[str, str | dict],
    use_gpu: bool,
) -> None:
    logger.info("Dataset: %s (%d samples)", dataset_name, dataset_size)
    logger.info("Loader variant: %s", "gpu" if use_gpu else "cpu")
    for name, entry in modality_info.items():
        if isinstance(entry, str):
            logger.info("  modality '%s': %s", name, entry)
        else:
            path = entry.get("path", "")
            split = entry.get("split")
            if split:
                logger.info("  modality '%s': %s (split=%s)", name, path, split)
            else:
                logger.info("  modality '%s': %s", name, path)


@contextmanager
def progress_bar(
    total: int, desc: str, logger: logging.Logger
) -> Iterator["tqdm | None"]:
    if tqdm is None:
        logger.info("%s: processing %d files", desc, total)
        yield None
        return
    bar = tqdm(total=total, desc=desc, unit="img")
    try:
        yield bar
    finally:
        bar.close()
