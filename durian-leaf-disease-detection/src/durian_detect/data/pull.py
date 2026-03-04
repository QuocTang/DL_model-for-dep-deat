"""Tải dữ liệu từ Roboflow."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from durian_detect.config import AppConfig

logger = logging.getLogger(__name__)


def _is_dir_empty(path: Path) -> bool:
    """Kiểm tra thư mục có rỗng hay không."""
    return path.exists() and not any(path.iterdir())


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

    # Roboflow SDK sẽ skip download nếu thư mục đích đã tồn tại.
    # → Nếu thư mục tồn tại nhưng rỗng, xóa đi để Roboflow tải lại.
    if _is_dir_empty(dst):
        logger.warning("Thư mục '%s' tồn tại nhưng rỗng, xóa để tải lại...", dst)
        dst.rmdir()

    # Chỉ tạo thư mục CHA, KHÔNG tạo thư mục đích (để Roboflow tự tạo khi download)
    dst.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Connecting to Roboflow workspace '%s'...", rf_cfg.workspace)
    logger.info("Project: '%s', Version: %d", rf_cfg.project, rf_cfg.version)
    logger.info("Download location: '%s'", dst)

    try:
        from roboflow import Roboflow

        rf = Roboflow(api_key=rf_cfg.api_key)
        project = rf.workspace(rf_cfg.workspace).project(rf_cfg.project)
        version = project.version(rf_cfg.version)
        dataset = version.download(rf_cfg.format, location=str(dst))
    except Exception as e:
        raise RuntimeError(f"Failed to download dataset: {e}") from e

    # Kiểm tra kết quả download
    if not dst.exists():
        raise RuntimeError(
            f"Download có vẻ thành công nhưng thư mục '{dst}' không tồn tại."
        )

    file_count = sum(1 for _ in dst.rglob("*") if _.is_file())
    if file_count == 0:
        logger.warning("⚠️  Thư mục '%s' tồn tại nhưng không có file nào!", dst)
    else:
        logger.info("✅ Dataset downloaded to: %s (%d files)", dst, file_count)

    return dst
