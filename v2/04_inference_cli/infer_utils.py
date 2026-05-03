"""
Inference utilities for deep-feature + classical ML prediction.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "02_feature_extraction"))
from config import DEFAULT_ML_MODEL, DEFAULT_RESULTS_JSON, DEFAULT_WEIGHTS
from extract_deep_features import YoloFeatureExtractor


DEFAULT_YOLO_WEIGHTS = DEFAULT_WEIGHTS


class DeepFeatureMLPredictor:
    def __init__(
        self,
        yolo_weights: Path,
        ml_model_path: Path,
        results_json_path: Path,
        layers: List[int],
        imgsz: int = 640,
    ):
        self.yolo_weights = yolo_weights
        self.ml_model_path = ml_model_path
        self.results_json_path = results_json_path
        self.layers = layers
        self.imgsz = imgsz

        self.extractor = YoloFeatureExtractor(yolo_weights, layers, input_size=imgsz)
        self.model = joblib.load(ml_model_path)

        with open(results_json_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        class_mapping_raw = results.get("class_mapping", {})
        raw_to_model = results.get("label_mapping_raw_to_model", {})

        # model_to_class maps contiguous model class IDs -> class names.
        self.model_to_class: Dict[int, str] = {}
        for raw_label, model_label in raw_to_model.items():
            raw_key = str(raw_label)
            class_name = class_mapping_raw.get(raw_key, f"class_{raw_key}")
            self.model_to_class[int(model_label)] = class_name

        if not self.model_to_class:
            raise RuntimeError("Cannot build class mapping from results.json")

    def predict_from_rgb(self, image_rgb: np.ndarray, top_k: int = 5) -> Dict:
        feat = self.extractor.extract_array(image_rgb)
        feat = feat.astype(np.float32).reshape(1, -1)

        pred_model_label = int(self.model.predict(feat)[0])

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(feat)[0]
        else:
            # Fallback: one-hot like probability when predict_proba is unavailable.
            max_idx = max(self.model_to_class.keys())
            probs = np.zeros(max_idx + 1, dtype=np.float32)
            probs[pred_model_label] = 1.0

        rows = []
        for idx, prob in enumerate(probs):
            class_name = self.model_to_class.get(idx, f"class_{idx}")
            rows.append({"model_label": idx, "class_name": class_name, "probability": float(prob)})

        rows.sort(key=lambda x: x["probability"], reverse=True)
        top_rows = rows[:top_k]

        return {
            "predicted_model_label": pred_model_label,
            "predicted_class": self.model_to_class.get(pred_model_label, f"class_{pred_model_label}"),
            "top_k": top_rows,
        }
