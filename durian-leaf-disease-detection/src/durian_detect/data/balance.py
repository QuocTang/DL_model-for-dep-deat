"""Cân bằng dữ liệu bằng Downsampling các lớp chiếm đa số."""

from __future__ import annotations

import logging
import random
import shutil
from pathlib import Path

from durian_detect.config import AppConfig

logger = logging.getLogger(__name__)


def downsample(config: AppConfig) -> Path:
    """Downsample các lớp có quá nhiều mẫu.

    Quy trình:
        1. Tìm tất cả ảnh chứa target classes.
        2. Random chọn keep_ratio% ảnh giữ lại.
        3. Copy ảnh + nhãn đã chọn + toàn bộ ảnh các lớp khác.

    Args:
        config: Cấu hình ứng dụng.

    Returns:
        Đường dẫn tới thư mục dữ liệu đã cân bằng.
    """
    src_root = Path(config.paths.refactored_data)
    dst_root = Path(config.paths.balanced_data)
    target_classes = set(config.balance.target_classes)
    keep_ratio = config.balance.keep_ratio
    seed = config.balance.seed

    if not src_root.exists():
        raise FileNotFoundError(f"Source data not found: {src_root}")

    random.seed(seed)

    logger.info("Downsampling classes %s with keep_ratio=%.1f%%", target_classes, keep_ratio * 100)

    for split in config.balance.splits:
        image_dir = src_root / split / "images"
        label_dir = src_root / split / "labels"

        if not image_dir.exists():
            logger.warning("Split '%s' not found, skipping.", split)
            continue

        new_image_dir = dst_root / split / "images"
        new_label_dir = dst_root / split / "labels"
        new_image_dir.mkdir(parents=True, exist_ok=True)
        new_label_dir.mkdir(parents=True, exist_ok=True)

        # --- Tìm ảnh chứa target classes ---
        all_images = list(image_dir.iterdir())
        large_class_images: set[str] = set()

        for img_path in all_images:
            label_path = label_dir / img_path.with_suffix(".txt").name
            if not label_path.exists():
                continue

            for line in label_path.read_text(encoding="utf-8").strip().splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                class_id = int(parts[0])
                if class_id in target_classes:
                    large_class_images.add(img_path.name)
                    break

        # --- Random chọn ảnh giữ lại ---
        keep_count = int(len(large_class_images) * keep_ratio)
        keep_images = set(random.sample(sorted(large_class_images), keep_count))

        logger.info(
            "  [%s] Found %d images with large classes → keeping %d (%.0f%%)",
            split,
            len(large_class_images),
            keep_count,
            keep_ratio * 100,
        )

        # --- Copy ảnh: giữ nguyên ảnh lớp nhỏ + random ảnh lớp lớn ---
        copied = 0
        skipped = 0

        for img_path in all_images:
            if img_path.name in large_class_images and img_path.name not in keep_images:
                skipped += 1
                continue

            # Copy image
            shutil.copy2(img_path, new_image_dir / img_path.name)

            # Copy label
            label_path = label_dir / img_path.with_suffix(".txt").name
            if label_path.exists():
                shutil.copy2(label_path, new_label_dir / label_path.name)

            copied += 1

        logger.info(
            "  [%s] Result: %d copied, %d skipped", split, copied, skipped
        )

    logger.info("✅ Downsampling complete! Output: %s", dst_root)
    return dst_root
