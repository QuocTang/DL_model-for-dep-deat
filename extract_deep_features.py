"""
Extract deep features from a trained YOLO model for ML training.

This script only generates feature CSV files (train/valid/test) and does not train ML.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm
from ultralytics import YOLO


WORKSPACE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = WORKSPACE_DIR / "runs" / "detect" / "durian-1-yolo11x" / "weights" / "best.pt"
DEFAULT_DATA_YAML = WORKSPACE_DIR.parent / "CI_CO-10" / "data.yaml"
DEFAULT_DATASET_ROOT = WORKSPACE_DIR / "durian-1"
DEFAULT_OUTPUT_DIR = WORKSPACE_DIR / "extracted_features"
DEFAULT_LAYERS = [4, 6, 9]


def resolve_split_dir(data_yaml_path: Path, split_value: str) -> Path:
    yaml_dir = data_yaml_path.parent
    split_path = Path(split_value)
    candidates = []

    if split_path.is_absolute():
        candidates.append(split_path)

    candidates.append((yaml_dir / split_path).resolve())

    if split_value.startswith("../"):
        candidates.append((yaml_dir / split_value.replace("../", "", 1)).resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Cannot resolve split directory from '{split_value}'. Tried: {candidates}"
    )


class YoloFeatureExtractor:
    def __init__(self, model_path: Path, layer_indices: List[int], input_size: int = 640):
        self.model_path = model_path
        self.layer_indices = layer_indices
        self.input_size = input_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.yolo = YOLO(str(model_path))
        self.yolo.model.to(self.device)
        self.features: Dict[str, torch.Tensor] = {}

        self._register_hooks()

    def _register_hooks(self) -> None:
        def get_hook(name: str):
            def hook(module, inputs, output):
                self.features[name] = output.detach()

            return hook

        layers = self.yolo.model.model
        max_idx = len(layers) - 1

        for idx in self.layer_indices:
            if idx < 0 or idx > max_idx:
                raise ValueError(f"Invalid layer index {idx}. Model has layers 0..{max_idx}")
            layers[idx].register_forward_hook(get_hook(f"layer_{idx}"))

    def _letterbox(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        ratio = self.input_size / max(h, w)
        new_w, new_h = int(w * ratio), int(h * ratio)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        dw = self.input_size - new_w
        dh = self.input_size - new_h
        left = dw // 2
        right = dw - left
        top = dh // 2
        bottom = dh - top

        return cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

    def extract_array(self, image_rgb: np.ndarray) -> np.ndarray:
        if image_rgb is None or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("Input image must be an RGB array with shape HxWx3")

        image = image_rgb
        image = self._letterbox(image)

        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)

        self.features = {}
        with torch.no_grad():
            _ = self.yolo.model(tensor)

        vectors = []
        for key in sorted(self.features.keys()):
            feat = self.features[key]
            pooled = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1))
            vectors.append(pooled.flatten().cpu().numpy())

        if not vectors:
            raise RuntimeError("No intermediate features captured. Check layer indices.")

        return np.concatenate(vectors)

    def extract_one(self, image_path: Path) -> np.ndarray:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.extract_array(image)


def read_names(data_cfg: Dict) -> List[str]:
    names = data_cfg["names"]
    if isinstance(names, list):
        return names
    if isinstance(names, dict):
        return [names[k] for k in sorted(names.keys())]
    raise ValueError("Invalid names format in data.yaml")


def find_split_dir(root: Path, split_name: str) -> Path:
    candidates = [
        root / split_name,
        root / split_name.lower(),
        root / split_name.upper(),
    ]

    if split_name == "valid":
        candidates.extend([root / "val", root / "VAL"])
    if split_name == "val":
        candidates.extend([root / "valid", root / "VALID"])

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate

    raise FileNotFoundError(f"Cannot find split directory for '{split_name}' under: {root}")


def collect_image_paths(image_dir: Path) -> List[Path]:
    image_paths = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        image_paths.extend(image_dir.glob(pattern))
    return sorted(image_paths)


def load_tf_split_labels(split_dir: Path) -> Tuple[Dict[str, str], List[str]]:
    ann_file = split_dir / "_annotations.csv"
    if not ann_file.exists():
        raise FileNotFoundError(f"Missing annotation file: {ann_file}")

    df = pd.read_csv(ann_file)
    required_cols = {"filename", "class"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Annotation CSV must contain columns {required_cols}. Found: {list(df.columns)}"
        )

    # Keep only rows where image file is present to avoid stale labels.
    image_names = {p.name for p in collect_image_paths(split_dir)}
    df = df[df["filename"].isin(image_names)]
    if df.empty:
        return {}, []

    label_by_image: Dict[str, str] = {}
    class_order: List[str] = []

    for image_name, group in df.groupby("filename", sort=False):
        counts = group["class"].value_counts()
        dominant_class = str(counts.index[0])
        label_by_image[image_name] = dominant_class
        if dominant_class not in class_order:
            class_order.append(dominant_class)

    return label_by_image, class_order


def extract_split(
    extractor: YoloFeatureExtractor,
    data_yaml_path: Path,
    split_key: str,
    output_name: str,
    output_dir: Path,
    max_images: int | None = None,
) -> Path:
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    if split_key not in data_cfg:
        raise KeyError(f"Split '{split_key}' not present in {data_yaml_path}")

    class_names = read_names(data_cfg)
    image_dir = resolve_split_dir(data_yaml_path, data_cfg[split_key])
    label_dir = image_dir.parent / "labels"

    image_paths = collect_image_paths(image_dir)

    if max_images is not None:
        image_paths = image_paths[:max_images]

    print(f"\n[{split_key}] images: {len(image_paths)} from {image_dir}")

    rows = []
    for img_path in tqdm(image_paths, desc=f"Extracting {split_key}"):
        label_path = label_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue

        with open(label_path, "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f if line.strip()]

        if not labels:
            continue

        class_ids = [int(line.split()[0]) for line in labels]
        dominant_class = max(set(class_ids), key=class_ids.count)

        try:
            feat = extractor.extract_one(img_path)
        except Exception as exc:
            print(f"[WARN] skip {img_path.name}: {exc}")
            continue

        row = {f"feature_{i}": float(v) for i, v in enumerate(feat)}
        row["label"] = dominant_class
        row["class_name"] = class_names[dominant_class]
        row["image_name"] = img_path.name
        rows.append(row)

    if not rows:
        raise RuntimeError(f"No features extracted for split {split_key}")

    df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / f"{output_name}_features.csv"
    df.to_csv(out_csv, index=False)

    print(f"Saved: {out_csv} | shape={df.shape}")
    print(df["class_name"].value_counts())
    return out_csv


def extract_split_tensorflow(
    extractor: YoloFeatureExtractor,
    dataset_root: Path,
    split_dir_name: str,
    output_name: str,
    output_dir: Path,
    class_to_id: Dict[str, int],
    max_images: int | None = None,
) -> Path:
    split_dir = find_split_dir(dataset_root, split_dir_name)
    image_paths = collect_image_paths(split_dir)
    labels_by_image, _ = load_tf_split_labels(split_dir)

    if max_images is not None:
        image_paths = image_paths[:max_images]

    print(f"\n[{split_dir_name}] images: {len(image_paths)} from {split_dir}")

    rows = []
    for img_path in tqdm(image_paths, desc=f"Extracting {split_dir_name}"):
        class_name = labels_by_image.get(img_path.name)
        if class_name is None:
            continue

        try:
            feat = extractor.extract_one(img_path)
        except Exception as exc:
            print(f"[WARN] skip {img_path.name}: {exc}")
            continue

        row = {f"feature_{i}": float(v) for i, v in enumerate(feat)}
        row["label"] = class_to_id[class_name]
        row["class_name"] = class_name
        row["image_name"] = img_path.name
        rows.append(row)

    if not rows:
        raise RuntimeError(f"No features extracted for split {split_dir_name}")

    df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / f"{output_name}_features.csv"
    df.to_csv(out_csv, index=False)

    print(f"Saved: {out_csv} | shape={df.shape}")
    print(df["class_name"].value_counts())
    return out_csv


def build_tf_class_mapping(dataset_root: Path) -> Dict[str, int]:
    class_names: List[str] = []
    for split_name in ("train", "valid", "test"):
        try:
            split_dir = find_split_dir(dataset_root, split_name)
        except FileNotFoundError:
            continue

        _, split_classes = load_tf_split_labels(split_dir)
        for cls in split_classes:
            if cls not in class_names:
                class_names.append(cls)

    if not class_names:
        raise RuntimeError(f"Cannot find class names from TensorFlow annotations in {dataset_root}")

    # Deterministic class ID mapping.
    class_names = sorted(class_names)
    return {name: idx for idx, name in enumerate(class_names)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract deep features from trained YOLO model")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to best.pt")
    parser.add_argument("--data-yaml", type=Path, default=DEFAULT_DATA_YAML, help="Path to data.yaml")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Dataset root for TensorFlow export (contains train/valid/test + _annotations.csv)",
    )
    parser.add_argument(
        "--dataset-format",
        type=str,
        default="auto",
        choices=["auto", "yolo", "tensorflow"],
        help="Dataset format to parse",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for CSV")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--layers", type=str, default="4,6,9", help="Comma-separated YOLO layer indexes")
    parser.add_argument("--max-images", type=int, default=None, help="Limit images per split for test run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_path = args.model.resolve()
    data_yaml = args.data_yaml.resolve()
    dataset_root = args.dataset_root.resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    dataset_format = args.dataset_format
    if dataset_format == "auto":
        if (dataset_root / "train" / "_annotations.csv").exists() or (
            dataset_root / "valid" / "_annotations.csv"
        ).exists():
            dataset_format = "tensorflow"
        elif data_yaml.exists():
            dataset_format = "yolo"
        else:
            raise FileNotFoundError(
                "Cannot auto-detect dataset format. Provide --dataset-format and valid data path."
            )

    if dataset_format == "yolo" and not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")
    if dataset_format == "tensorflow" and not dataset_root.exists():
        raise FileNotFoundError(f"dataset root not found: {dataset_root}")

    layer_indices = [int(x.strip()) for x in args.layers.split(",") if x.strip()]

    print("=" * 80)
    print("DEEP FEATURE EXTRACTION")
    print("=" * 80)
    print(f"Model: {model_path}")
    if dataset_format == "yolo":
        print(f"Data : {data_yaml}")
    else:
        print(f"Data : {dataset_root}")
    print(f"Fmt  : {dataset_format}")
    print(f"Out  : {args.output_dir.resolve()}")
    print(f"Layer: {layer_indices}")
    print(f"CUDA : {torch.cuda.is_available()} ({torch.cuda.device_count()} devices)")

    extractor = YoloFeatureExtractor(model_path, layer_indices, input_size=args.imgsz)

    if dataset_format == "yolo":
        extract_split(extractor, data_yaml, "train", "train", args.output_dir, args.max_images)
        extract_split(extractor, data_yaml, "val", "valid", args.output_dir, args.max_images)
        extract_split(extractor, data_yaml, "test", "test", args.output_dir, args.max_images)
    else:
        class_to_id = build_tf_class_mapping(dataset_root)
        print(f"Classes: {class_to_id}")
        extract_split_tensorflow(
            extractor,
            dataset_root,
            "train",
            "train",
            args.output_dir,
            class_to_id,
            args.max_images,
        )
        extract_split_tensorflow(
            extractor,
            dataset_root,
            "valid",
            "valid",
            args.output_dir,
            class_to_id,
            args.max_images,
        )
        extract_split_tensorflow(
            extractor,
            dataset_root,
            "test",
            "test",
            args.output_dir,
            class_to_id,
            args.max_images,
        )

    print("\nDone: feature CSV files generated for train/valid/test.")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    main()
