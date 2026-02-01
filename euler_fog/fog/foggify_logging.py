import logging
from contextlib import contextmanager
from typing import Iterator

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def get_logger(name: str = "foggify") -> logging.Logger:
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


def log_config(
    logger: logging.Logger,
    config: dict,
    config_path: str,
    out_path: str,
    device: str,
    use_gpu: bool,
    torch_available: bool,
    torch_device: str | None,
    gpu_batch_size: int,
    file_step: int,
    depth_scale: float,
    resize_depth: bool,
    seed,
    contrast_threshold_default: float,
) -> None:
    logger.info("Loaded config from %s -> output %s", config_path, out_path)
    logger.info(
        "Device: requested=%s active=%s use_gpu=%s torch=%s",
        device,
        torch_device or "cpu",
        use_gpu,
        "available" if torch_available else "missing",
    )
    logger.info(
        "Processing: file_step=%s gpu_batch_size=%s depth_scale=%s resize_depth=%s seed=%s contrast_threshold=%s",
        file_step,
        gpu_batch_size,
        depth_scale,
        resize_depth,
        seed,
        contrast_threshold_default,
    )
    selection = config.get("selection")
    if selection:
        mode = selection.get("mode", "fixed")
        logger.info("Model selection: mode=%s", mode)
        if mode == "weighted":
            logger.info("Model weights: %s", selection.get("weights", {}))
        elif mode == "fixed":
            logger.info("Model fixed: %s", selection.get("model", "uniform"))


def log_dataset_info(
    logger: logging.Logger,
    dataset_name: str,
    dataset_size: int,
    modality_paths: dict[str, str],
    use_gpu: bool,
) -> None:
    logger.info("Dataset: %s (%d samples)", dataset_name, dataset_size)
    logger.info("Loader variant: %s", "gpu" if use_gpu else "cpu")
    for name, path in modality_paths.items():
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
