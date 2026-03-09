"""Tái cấu trúc nhãn (labels) — gộp/ánh xạ class ID."""

from __future__ import annotations

import logging
import shutil
from collections import Counter
from pathlib import Path

from durian_detect.config import AppConfig

logger = logging.getLogger(__name__)

SPLITS = ("train", "valid", "test")


def _remap_class_id(old_id: int, mapping: dict[int, int], default: int) -> int:
    """Ánh xạ class ID cũ sang mới."""
    return mapping.get(old_id, default)


def refactor_labels(config: AppConfig) -> Path:
    """Tái cấu trúc nhãn: giữ các lớp chính, gộp còn lại.

    Quy trình:
        1. Duyệt qua train/valid/test splits.
        2. Copy ảnh nguyên vẹn sang thư mục đích.
        3. Đọc file nhãn, ánh xạ class ID, ghi file nhãn mới.

    Args:
        config: Cấu hình ứng dụng.

    Returns:
        Đường dẫn tới thư mục dữ liệu đã refactor.
    """
    src_root = Path(config.paths.raw_data)
    dst_root = Path(config.paths.refactored_data)
    mapping = config.class_mapping.mapping
    default_id = config.class_mapping.default

    if not src_root.exists():
        raise FileNotFoundError(f"Source data not found: {src_root}")

    logger.info("Refactoring labels: %s → %s", src_root, dst_root)
    logger.info("Class mapping: %s (default → %d)", mapping, default_id)

    total_images = 0
    total_labels = 0
    class_counter: Counter[int] = Counter()

    for split in SPLITS:
        img_src = src_root / split / "images"
        lbl_src = src_root / split / "labels"

        if not img_src.exists():
            logger.warning("Split '%s' not found, skipping.", split)
            continue

        img_dst = dst_root / split / "images"
        lbl_dst = dst_root / split / "labels"
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        # Copy images
        image_files = list(img_src.iterdir())
        for img_file in image_files:
            shutil.copy2(img_file, img_dst / img_file.name)
        total_images += len(image_files)

        # Rewrite labels
        if not lbl_src.exists():
            logger.warning("Labels dir not found for split '%s'.", split)
            continue

        label_files = list(lbl_src.iterdir())
        for lbl_file in label_files:
            lines = lbl_file.read_text(encoding="utf-8").strip().splitlines()
            new_lines: list[str] = []

            for line in lines:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue

                old_id = int(parts[0])
                new_id = _remap_class_id(old_id, mapping, default_id)
                parts[0] = str(new_id)
                new_lines.append(" ".join(parts))
                class_counter[new_id] += 1

            (lbl_dst / lbl_file.name).write_text(
                "\n".join(new_lines), encoding="utf-8"
            )
            total_labels += 1

        logger.info(
            "  [%s] %d images, %d label files processed.",
            split,
            len(image_files),
            len(label_files),
        )

    logger.info("✅ Refactoring complete!")
    logger.info("   Total: %d images, %d label files", total_images, total_labels)
    logger.info("   Class distribution (new IDs): %s", dict(class_counter.most_common()))

    return dst_root
