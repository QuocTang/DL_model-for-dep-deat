"""Configuration loader & dataclass definitions."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# ── Dataclasses ──────────────────────────────────────────────


@dataclass
class RoboflowConfig:
    """Cấu hình kết nối Roboflow."""

    api_key: str = ""
    workspace: str = ""
    project: str = ""
    version: int = 1
    format: str = "yolov11"


@dataclass
class PathsConfig:
    """Đường dẫn dữ liệu."""

    raw_data: str = "./data/raw"
    refactored_data: str = "./data/refactored"
    balanced_data: str = "./data/balanced"
    output: str = "./outputs"


@dataclass
class ClassMappingConfig:
    """Ánh xạ class ID cũ → mới."""

    mapping: dict[int, int] = field(default_factory=dict)
    default: int = 5


@dataclass
class BalanceConfig:
    """Cấu hình cân bằng dữ liệu."""

    target_classes: list[int] = field(default_factory=lambda: [2, 12])
    keep_ratio: float = 0.4
    splits: list[str] = field(default_factory=lambda: ["train"])
    seed: int = 42


@dataclass
class TrainingConfig:
    """Hyperparameters cho huấn luyện."""

    model: str = "yolo11x.pt"
    data_yaml: str | None = None
    batch: int = 8
    epochs: int = 100
    device: int | str = 0
    imgsz: int = 640
    optimizer: str = "AdamW"
    lr0: float = 0.005
    lrf: float = 0.10
    weight_decay: float = 0.0005
    warmup_epochs: int = 3
    cos_lr: bool = True
    patience: int = 50
    seed: int = 42
    pretrained: bool = True
    save: bool = True
    val: bool = True
    plots: bool = True
    name: str = "durian-1-yolo11x"


@dataclass
class VisualizationConfig:
    """Cấu hình trực quan hóa."""

    splits: list[str] = field(default_factory=lambda: ["train", "valid"])
    figsize: list[int] = field(default_factory=lambda: [14, 6])
    save_plot: bool = True


@dataclass
class AppConfig:
    """Cấu hình toàn bộ ứng dụng."""

    roboflow: RoboflowConfig = field(default_factory=RoboflowConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    class_mapping: ClassMappingConfig = field(default_factory=ClassMappingConfig)
    balance: BalanceConfig = field(default_factory=BalanceConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    original_class_names: list[str] = field(default_factory=list)
    refactored_class_names: list[str] = field(default_factory=list)


# ── Helpers ──────────────────────────────────────────────────


def _build_dataclass(cls: type, data: dict[str, Any] | None):
    """Tạo dataclass instance từ dict, bỏ qua keys không hợp lệ."""
    if data is None:
        return cls()
    valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in valid_fields}
    return cls(**filtered)


def _resolve_path(base: Path, p: str) -> str:
    """Chuyển đường dẫn tương đối thành tuyệt đối dựa trên base dir."""
    path = Path(p)
    if not path.is_absolute():
        path = base / path
    return str(path.resolve())


# ── Main loader ──────────────────────────────────────────────


def load_config(config_path: str | Path) -> AppConfig:
    """Đọc file YAML và trả về AppConfig.

    Args:
        config_path: Đường dẫn tới file YAML.

    Returns:
        AppConfig instance đã validate.

    Raises:
        FileNotFoundError: Nếu file config không tồn tại.
        ValueError: Nếu file config không hợp lệ.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    logger.info("Loaded config from %s", config_path)

    # --- Roboflow ---
    rf_data = raw.get("roboflow", {})
    # Ưu tiên biến môi trường cho API key
    env_key = os.environ.get("ROBOFLOW_API_KEY")
    if env_key:
        rf_data["api_key"] = env_key
        logger.info("Using ROBOFLOW_API_KEY from environment variable")
    roboflow = _build_dataclass(RoboflowConfig, rf_data)

    # --- Paths (resolve relative paths) ---
    base_dir = config_path.parent.parent  # project root
    paths_data = raw.get("paths", {})
    for key in paths_data:
        paths_data[key] = _resolve_path(base_dir, paths_data[key])
    paths = _build_dataclass(PathsConfig, paths_data)

    # --- Class mapping ---
    cm_raw = raw.get("class_mapping", {})
    default_id = cm_raw.pop("default", 5)
    mapping = {int(k): int(v) for k, v in cm_raw.items()}
    class_mapping = ClassMappingConfig(mapping=mapping, default=default_id)

    # --- Balance ---
    balance = _build_dataclass(BalanceConfig, raw.get("balance"))

    # --- Training ---
    training = _build_dataclass(TrainingConfig, raw.get("training"))

    # --- Visualization ---
    visualization = _build_dataclass(VisualizationConfig, raw.get("visualization"))

    # --- Class names ---
    original_class_names = raw.get("original_class_names", [])
    refactored_class_names = raw.get("refactored_class_names", [])

    config = AppConfig(
        roboflow=roboflow,
        paths=paths,
        class_mapping=class_mapping,
        balance=balance,
        training=training,
        visualization=visualization,
        original_class_names=original_class_names,
        refactored_class_names=refactored_class_names,
    )

    logger.debug("Resolved config: %s", config)
    return config
