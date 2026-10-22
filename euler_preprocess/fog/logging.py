from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any


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
    scenarios: Sequence[tuple[str, Any, Any]] = (),
    scenario_scope: str = "sample",
) -> None:
    """Log the resolved run settings.

    *scenarios* holds ``(name, weight, model)`` per scenario profile, with
    ``model`` ``None`` when the profile defers to ``selection``.
    """
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
    if scenarios:
        logger.info(
            "Model selection: %d scenario_profiles (scope=%s) choose the model; "
            "'selection' only applies to profiles without one",
            len(scenarios),
            scenario_scope,
        )
        for name, weight, model in scenarios:
            logger.info(
                "Scenario %s: weight=%s model=%s",
                name,
                weight,
                "<selection>" if model is None else model,
            )
        if all(model is not None for _name, _weight, model in scenarios):
            return
    selection = config.get("selection")
    if selection:
        mode = selection.get("mode", "fixed")
        logger.info("Model selection: mode=%s", mode)
        if mode == "weighted":
            logger.info("Model weights: %s", selection.get("weights", {}))
        elif mode == "fixed":
            logger.info("Model fixed: %s", selection.get("model", "uniform"))
