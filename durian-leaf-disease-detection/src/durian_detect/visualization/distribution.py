"""Trực quan hóa phân phối lớp bệnh."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from durian_detect.config import AppConfig

logger = logging.getLogger(__name__)


def plot_distribution(
    config: AppConfig,
    data_root: str | Path | None = None,
    class_names: list[str] | None = None,
) -> Path | None:
    """Vẽ biểu đồ phân phối lớp cho train/valid splits.

    Args:
        config: Cấu hình ứng dụng.
        data_root: Đường dẫn dataset (mặc định: config.paths.raw_data).
        class_names: Tên lớp (mặc định: config.original_class_names).

    Returns:
        Đường dẫn file biểu đồ nếu save_plot=True, None nếu hiển thị trực tiếp.
    """
    root = Path(data_root or config.paths.raw_data)
    names = class_names or config.original_class_names
    splits = config.visualization.splits
    figsize = tuple(config.visualization.figsize)

    if not root.exists():
        raise FileNotFoundError(f"Data root not found: {root}")

    if not names:
        raise ValueError("class_names is empty. Check config or provide manually.")

    logger.info("Plotting class distribution from: %s", root)
    logger.info("Splits: %s, Classes: %d", splits, len(names))

    records: list[dict] = []

    for split in splits:
        label_dir = root / split / "labels"
        if not label_dir.exists():
            logger.warning("Label dir not found: %s", label_dir)
            continue

        counts = [0] * len(names)

        for label_file in label_dir.iterdir():
            if label_file.stat().st_size == 0:
                continue

            for line in label_file.read_text(encoding="utf-8").strip().splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                class_id = int(parts[0])
                if 0 <= class_id < len(names):
                    counts[class_id] += 1

        for i, count in enumerate(counts):
            records.append({"Split": split, "Class": names[i], "Count": count})

        logger.info("  [%s] Total annotations: %d", split, sum(counts))

    if not records:
        logger.warning("No data to plot!")
        return None

    df = pd.DataFrame(records)

    # --- Vẽ biểu đồ ---
    plt.figure(figsize=figsize)
    sns.barplot(data=df, x="Class", y="Count", hue="Split")
    plt.xticks(rotation=75)
    plt.title("Class Distribution — Train vs Valid")
    plt.tight_layout()

    if config.visualization.save_plot:
        output_dir = Path(config.paths.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / "class_distribution.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("✅ Plot saved to: %s", save_path)
        return save_path
    else:
        plt.show()
        return None
