"""Backward-compatibility shim.

Canonical locations:
- ``euler_preprocess.common.logging`` (get_logger, progress_bar, log_dataset_info)
- ``euler_preprocess.fog.logging`` (log_config)
"""
from __future__ import annotations

from euler_preprocess.common.logging import get_logger, log_dataset_info, progress_bar  # noqa: F401
from euler_preprocess.fog.logging import log_config  # noqa: F401
