"""CLI entry point — argparse sub-commands cho toàn bộ pipeline."""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import platform
import sys
from pathlib import Path

from durian_detect.config import load_config


def _setup_logging(verbose: bool = False) -> None:
    """Cấu hình logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
        datefmt="%H:%M:%S",
    )


def _resolve_config_path(config_arg: str) -> Path:
    """Tìm đường dẫn config file."""
    p = Path(config_arg)
    if p.exists():
        return p

    # Thử tìm từ thư mục gốc project
    project_root = Path(__file__).parent.parent.parent
    candidate = project_root / config_arg
    if candidate.exists():
        return candidate

    raise FileNotFoundError(f"Config file not found: {config_arg}")


# ── Sub-command handlers ────────────────────────────────────


def cmd_pull(args: argparse.Namespace) -> None:
    """Tải dữ liệu từ Roboflow."""
    config = load_config(_resolve_config_path(args.config))

    from durian_detect.data.pull import pull_dataset

    pull_dataset(config)


def cmd_refactor(args: argparse.Namespace) -> None:
    """Tái cấu trúc nhãn."""
    config = load_config(_resolve_config_path(args.config))

    from durian_detect.data.refactor import refactor_labels

    refactor_labels(config)


def cmd_balance(args: argparse.Namespace) -> None:
    """Cân bằng dữ liệu."""
    config = load_config(_resolve_config_path(args.config))

    from durian_detect.data.balance import downsample

    downsample(config)


def cmd_plot(args: argparse.Namespace) -> None:
    """Vẽ biểu đồ phân phối lớp."""
    config = load_config(_resolve_config_path(args.config))

    from durian_detect.visualization.distribution import plot_distribution

    plot_distribution(config)


def cmd_train(args: argparse.Namespace) -> None:
    """Huấn luyện model."""
    config = load_config(_resolve_config_path(args.config))

    from durian_detect.training.train import train_model

    train_model(config)


def cmd_pipeline(args: argparse.Namespace) -> None:
    """Chạy toàn bộ pipeline từ đầu đến cuối."""
    config = load_config(_resolve_config_path(args.config))
    logger = logging.getLogger(__name__)

    steps = [
        ("1/5 — Pull data", "durian_detect.data.pull", "pull_dataset"),
        ("2/5 — Refactor labels", "durian_detect.data.refactor", "refactor_labels"),
        ("3/5 — Balance data", "durian_detect.data.balance", "downsample"),
        ("4/5 — Plot distribution", "durian_detect.visualization.distribution", "plot_distribution"),
        ("5/5 — Train model", "durian_detect.training.train", "train_model"),
    ]

    skip = set()
    if args.skip:
        skip = set(args.skip)
        logger.info("Skipping steps: %s", skip)

    import importlib

    for label, module_path, func_name in steps:
        step_id = label.split(" — ")[1].lower().replace(" ", "_")
        if step_id in skip:
            logger.info("⏭️  Skipping %s", label)
            continue

        logger.info("=" * 60)
        logger.info("🚀 %s", label)
        logger.info("=" * 60)

        module = importlib.import_module(module_path)
        func = getattr(module, func_name)
        func(config)

    logger.info("=" * 60)
    logger.info("🎉 Pipeline complete!")
    logger.info("=" * 60)


# ── Main parser ─────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    """Xây dựng argument parser."""
    parser = argparse.ArgumentParser(
        prog="durian-detect",
        description="🍈 Durian Leaf Disease Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  uv run main.py pull                          # Tải dữ liệu từ Roboflow
  uv run main.py refactor                      # Tái cấu trúc nhãn
  uv run main.py balance                       # Cân bằng dữ liệu
  uv run main.py plot                          # Vẽ biểu đồ phân phối
  uv run main.py train                         # Huấn luyện model
  uv run main.py pipeline                      # Chạy toàn bộ
  uv run main.py pipeline --skip train         # Pipeline không train
  uv run main.py train --config config/custom.yaml
        """,
    )

    parser.add_argument(
        "--config", "-c",
        default="config/default.yaml",
        help="Đường dẫn tới file config YAML (mặc định: config/default.yaml)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Bật debug logging",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="Commands",
        description="Chọn bước cần thực hiện:",
    )

    # pull
    subparsers.add_parser("pull", help="Tải dữ liệu từ Roboflow")

    # refactor
    subparsers.add_parser("refactor", help="Tái cấu trúc nhãn (ánh xạ class ID)")

    # balance
    subparsers.add_parser("balance", help="Cân bằng dữ liệu (downsample)")

    # plot
    subparsers.add_parser("plot", help="Vẽ biểu đồ phân phối lớp")

    # train
    subparsers.add_parser("train", help="Huấn luyện YOLOv11x model")

    # pipeline
    pipeline_parser = subparsers.add_parser("pipeline", help="Chạy toàn bộ pipeline")
    pipeline_parser.add_argument(
        "--skip",
        nargs="+",
        choices=["pull_data", "refactor_labels", "balance_data", "plot_distribution", "train_model"],
        help="Bỏ qua các bước chỉ định",
    )

    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    _setup_logging(args.verbose)

    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Multiprocessing setup (cần thiết trên Windows)
    if platform.system() == "Windows":
        multiprocessing.freeze_support()
        multiprocessing.set_start_method("spawn", force=True)

    # Dispatch
    handlers = {
        "pull": cmd_pull,
        "refactor": cmd_refactor,
        "balance": cmd_balance,
        "plot": cmd_plot,
        "train": cmd_train,
        "pipeline": cmd_pipeline,
    }

    handler = handlers.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
