"""Huấn luyện YOLOv11x model."""

from __future__ import annotations

import logging
import platform
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

from durian_detect.config import AppConfig

logger = logging.getLogger(__name__)


def _log_gpu_info() -> None:
    """In thông tin GPU/CUDA."""
    logger.info("=" * 50)
    logger.info("System: %s %s", platform.system(), platform.machine())
    logger.info("Python: %s", sys.version.split()[0])
    logger.info("PyTorch: %s", torch.__version__)
    logger.info("CUDA available: %s", torch.cuda.is_available())

    if torch.cuda.is_available():
        logger.info("CUDA version: %s", torch.version.cuda)
        logger.info("GPU count: %d", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_mem / (1024**3)
            logger.info("  GPU %d: %s (%.1f GB)", i, name, mem)
            logger.info("  GPU %d: %s (%.1f GB)", i, name, mem)
    elif torch.backends.mps.is_available():
        logger.info("Apple Silicon MPS (Metal Performance Shaders) detected!")
        logger.info("Training will run on MAC GPU.")
    else:
        logger.warning("No GPU (CUDA/MPS) detected! Training will run on CPU (very slow).")

    logger.info("=" * 50)


def _generate_data_yaml(config: AppConfig) -> str:
    """Tự động tạo file dataset.yaml kết nối các tập dữ liệu."""
    out_yaml = Path(config.paths.balanced_data) / "dataset.yaml"
    out_yaml.parent.mkdir(parents=True, exist_ok=True)

    train_path = Path(config.paths.balanced_data) / "train" / "images"
    if not train_path.exists():
        logger.warning("Không tìm thấy folder balanced_data train. Dùng refactored_data thay thế.")
        train_path = Path(config.paths.refactored_data) / "train" / "images"

    val_path = Path(config.paths.refactored_data) / "valid" / "images"
    test_path = Path(config.paths.refactored_data) / "test" / "images"

    data = {
        "train": str(train_path.resolve()),
        "val": str(val_path.resolve()),
        "nc": len(config.refactored_class_names),
        "names": config.refactored_class_names,
    }
    
    if test_path.exists():
        data["test"] = str(test_path.resolve())

    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    logger.info("✅ Auto-generated data YAML at: %s", out_yaml)
    return str(out_yaml.resolve())


def train_model(config: AppConfig) -> Any:
    """Huấn luyện YOLOv11x model.

    Args:
        config: Cấu hình ứng dụng.

    Returns:
        Training results object từ Ultralytics.

    Raises:
        ValueError: Nếu data_yaml không được chỉ định.
        RuntimeError: Nếu training thất bại.
    """
    t_cfg = config.training

    if not t_cfg.data_yaml:
        logger.info("training.data_yaml chưa được cài đặt. Hệ thống sẽ tự động tạo file cấu hình dữ liệu...")
        data_yaml_path = _generate_data_yaml(config)
    else:
        data_yaml_path = t_cfg.data_yaml

    _log_gpu_info()

    logger.info("Loading model: %s (pretrained=%s)", t_cfg.model, t_cfg.pretrained)

    from ultralytics import YOLO

    model = YOLO(t_cfg.model)

    logger.info("Starting training...")
    logger.info("  Epochs: %d | Batch: %d | ImgSize: %d", t_cfg.epochs, t_cfg.batch, t_cfg.imgsz)
    logger.info("  Optimizer: %s | LR: %.4f → %.4f", t_cfg.optimizer, t_cfg.lr0, t_cfg.lr0 * t_cfg.lrf)
    logger.info("  Device: %s | Patience: %d", t_cfg.device, t_cfg.patience)
    logger.info("  Output: %s/%s", t_cfg.project or "runs/detect", t_cfg.name)

    try:
        results = model.train(
            data=data_yaml_path,
            batch=t_cfg.batch,
            epochs=t_cfg.epochs,
            device=t_cfg.device,
            imgsz=t_cfg.imgsz,
            optimizer=t_cfg.optimizer,
            lr0=t_cfg.lr0,
            lrf=t_cfg.lrf,
            weight_decay=t_cfg.weight_decay,
            warmup_epochs=t_cfg.warmup_epochs,
            cos_lr=t_cfg.cos_lr,
            patience=t_cfg.patience,
            seed=t_cfg.seed,
            pretrained=t_cfg.pretrained,
            save=t_cfg.save,
            val=t_cfg.val,
            plots=t_cfg.plots,
            name=t_cfg.name,
            project=t_cfg.project,
        )
    except Exception as e:
        raise RuntimeError(f"Training failed: {e}") from e

    logger.info("✅ Training complete!")

    # --- Validation riêng ---
    logger.info("Running final validation...")
    try:
        metrics = model.val(
            data=data_yaml_path,
            imgsz=t_cfg.imgsz,
            batch=t_cfg.batch,
            save_json=True,
            project=t_cfg.project,
            name=f"{t_cfg.name}_val",
        )
        logger.info("Validation metrics: %s", metrics)
    except Exception as e:
        logger.warning("Validation failed: %s", e)
        metrics = None

    return results, metrics
