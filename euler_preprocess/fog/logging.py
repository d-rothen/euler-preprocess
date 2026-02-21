from __future__ import annotations

import logging


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
