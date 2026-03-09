"""Trực quan hóa phân phối lớp bệnh."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from durian_detect.config import AppConfig

logger = logging.getLogger(__name__)


def _count_classes(
    root: Path,
    splits: list[str],
    num_classes: int,
) -> tuple[list[dict], int]:
    """Đếm annotations theo class cho từng split.

    Returns:
        (records, total_skipped)
    """
    records: list[dict] = []
    total_skipped = 0

    for split in splits:
        label_dir = root / split / "labels"
        if not label_dir.exists():
            logger.warning("Label dir not found: %s", label_dir)
            continue

        counts = [0] * num_classes
        skipped = 0

        for label_file in label_dir.iterdir():
            if label_file.stat().st_size == 0:
                continue

            for line in label_file.read_text(encoding="utf-8").strip().splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                class_id = int(parts[0])
                if 0 <= class_id < num_classes:
                    counts[class_id] += 1
                else:
                    skipped += 1

        for i, count in enumerate(counts):
            records.append({"Split": split, "Count": count, "_class_id": i})

        logger.info("  [%s] Total annotations: %d", split, sum(counts))
        if skipped:
            logger.warning(
                "  [%s] ⚠️ %d annotations bị bỏ qua (class_id ngoài range 0-%d)",
                split, skipped, num_classes - 1,
            )
            total_skipped += skipped

    return records, total_skipped


def _plot_single(
    df: pd.DataFrame,
    title: str,
    figsize: tuple,
    save_path: Path | None,
) -> Path | None:
    """Vẽ 1 biểu đồ barplot."""
    plt.figure(figsize=figsize)
    sns.barplot(data=df, x="Class", y="Count", hue="Split")
    plt.xticks(rotation=75)
    plt.title(title)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("✅ Plot saved to: %s", save_path)
        return save_path
    else:
        plt.show()
        return None


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

    records, _ = _count_classes(root, splits, len(names))

    if not records:
        logger.warning("No data to plot!")
        return None

    df = pd.DataFrame(records)
    df["Class"] = df["_class_id"].map(lambda i: names[i])
    df = df.drop(columns=["_class_id"])

    save_path = None
    if config.visualization.save_plot:
        output_dir = Path(config.paths.output)
        save_path = output_dir / "class_distribution.png"

    return _plot_single(df, "Class Distribution — Train vs Valid", figsize, save_path)


def plot_all_distributions(config: AppConfig) -> list[Path | None]:
    """Vẽ biểu đồ phân phối cho cả 3 dataset: raw, refactored, balanced.

    Returns:
        Danh sách đường dẫn file biểu đồ đã tạo.
    """
    splits = config.visualization.splits
    figsize = tuple(config.visualization.figsize)
    output_dir = Path(config.paths.output)
    save = config.visualization.save_plot

    datasets = [
        {
            "root": Path(config.paths.raw_data),
            "names": config.original_class_names,
            "title": "Class Distribution — Raw Data (15 lớp gốc)",
            "filename": "dist_raw.png",
        },
        {
            "root": Path(config.paths.refactored_data),
            "names": config.refactored_class_names,
            "title": "Class Distribution — Refactored Data (6 lớp)",
            "filename": "dist_refactored.png",
        },
        {
            "root": Path(config.paths.balanced_data),
            "names": config.refactored_class_names,
            "title": "Class Distribution — Balanced Data (sau downsample)",
            "filename": "dist_balanced.png",
        },
    ]

    results: list[Path | None] = []

    for ds in datasets:
        root = ds["root"]
        if not root.exists():
            logger.warning("⏭️  Bỏ qua '%s' — chưa có dữ liệu.", root)
            results.append(None)
            continue

        names = ds["names"]
        if not names:
            logger.warning("⏭️  Bỏ qua '%s' — thiếu class_names.", root)
            results.append(None)
            continue

        logger.info("=" * 50)
        logger.info("📊 %s", ds["title"])
        logger.info("   Source: %s", root)

        records, _ = _count_classes(root, splits, len(names))

        if not records:
            logger.warning("   No data to plot!")
            results.append(None)
            continue

        df = pd.DataFrame(records)
        df["Class"] = df["_class_id"].map(lambda i, n=names: n[i])
        df = df.drop(columns=["_class_id"])

        save_path = (output_dir / ds["filename"]) if save else None
        result = _plot_single(df, ds["title"], figsize, save_path)
        results.append(result)

    logger.info("=" * 50)
    logger.info("🎉 Done! %d biểu đồ đã tạo.", sum(1 for r in results if r is not None))
    return results
