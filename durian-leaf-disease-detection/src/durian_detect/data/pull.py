"""Tải dữ liệu từ Roboflow."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from durian_detect.config import AppConfig

logger = logging.getLogger(__name__)


def _has_data(directory: Path) -> bool:
    """Kiểm tra thư mục đã chứa dữ liệu (ít nhất 1 file) hay chưa."""
    if not directory.exists():
        return False
    return any(directory.rglob("*"))


def pull_dataset(config: AppConfig, *, force: bool = False) -> Path:
    """Tải dataset từ Roboflow về thư mục local.

    Args:
        config: Cấu hình ứng dụng.
        force: Nếu True, xóa dữ liệu cũ và tải lại từ đầu.

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

    if _has_data(dst):
        if not force:
            logger.warning(
                "⚠️  Thư mục '%s' đã chứa dữ liệu. "
                "Bỏ qua việc tải lại. Dùng --force để tải lại từ đầu.",
                dst,
            )
            return dst
        logger.info("🗑️  --force được bật. Xóa dữ liệu cũ tại '%s'...", dst)
        shutil.rmtree(dst)

    # Roboflow SDK sẽ skip download nếu thư mục đích đã tồn tại.
    # → Nếu thư mục tồn tại nhưng rỗng, xóa đi để Roboflow tải lại.
    if dst.exists() and not any(dst.iterdir()):
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

    if not _has_data(dst):
        raise RuntimeError(
            f"Download hoàn tất nhưng không tìm thấy dữ liệu tại '{dst}'. "
            "Vui lòng kiểm tra API key và tên project/version."
        )

    file_count = sum(1 for _ in dst.rglob("*") if _.is_file())
    logger.info("✅ Dataset downloaded to: %s (%d files)", dst, file_count)
    return dst
