"""
CLI inference for one image.

Example:
  python v2/06_inference/cli.py --image path/to/image.jpg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from core import (
    DEFAULT_ML_MODEL,
    DEFAULT_RESULTS_JSON,
    DEFAULT_YOLO_WEIGHTS,
    DeepFeatureMLPredictor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for one image")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--yolo", type=Path, default=DEFAULT_YOLO_WEIGHTS)
    parser.add_argument("--ml-model", type=Path, default=DEFAULT_ML_MODEL)
    parser.add_argument("--results-json", type=Path, default=DEFAULT_RESULTS_JSON)
    parser.add_argument("--layers", type=str, default="4,6,9")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    predictor = DeepFeatureMLPredictor(
        yolo_weights=args.yolo,
        ml_model_path=args.ml_model,
        results_json_path=args.results_json,
        layers=layers,
        imgsz=args.imgsz,
    )

    image = Image.open(args.image).convert("RGB")
    image_np = np.array(image, dtype=np.uint8)

    pred = predictor.predict_from_rgb(image_np, top_k=args.top_k)
    print(json.dumps(pred, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
