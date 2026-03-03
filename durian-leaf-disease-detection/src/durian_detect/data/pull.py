"""Tải dữ liệu từ Roboflow."""

from __future__ import annotations

import logging
from pathlib import Path

from durian_detect.config import AppConfig

logger = logging.getLogger(__name__)


def pull_dataset(config: AppConfig) -> Path:
    """Tải dataset từ Roboflow về thư mục local.

    Args:
        config: Cấu hình ứng dụng.

    Returns:
        Đường dẫn tới thư mục dataset đã tải.

    Raises:
        ValueError: Nếu thiếu API key.
        RuntimeError: Nếu tải thất bại.
    """
    rf_cfg = config.roboflow
    if not rf_cfg.api_key:
        raise ValueError(
            "Roboflow API key is required. "
            "Set it in config file or via ROBOFLOW_API_KEY environment variable."
        )

    dst = Path(config.paths.raw_data)
    dst.mkdir(parents=True, exist_ok=True)

    logger.info("Connecting to Roboflow workspace '%s'...", rf_cfg.workspace)
    logger.info("Project: '%s', Version: %d", rf_cfg.project, rf_cfg.version)

    try:
        from roboflow import Roboflow

        rf = Roboflow(api_key=rf_cfg.api_key)
        project = rf.workspace(rf_cfg.workspace).project(rf_cfg.project)
        version = project.version(rf_cfg.version)
        dataset = version.download(rf_cfg.format, location=str(dst))
    except Exception as e:
        raise RuntimeError(f"Failed to download dataset: {e}") from e

    logger.info("✅ Dataset downloaded to: %s", dst)
    return dst
