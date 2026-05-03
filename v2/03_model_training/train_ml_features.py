"""
Train classical ML on extracted YOLO deep features.

Default flow:
- Train on train_features.csv
- Use valid_features.csv for optional early diagnostics
- Report final metrics on test_features.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import EXTRACTED_FEATURES_DIR, ML_MODELS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ML model on extracted deep features")
    parser.add_argument("--features-dir", type=Path, default=EXTRACTED_FEATURES_DIR)
    parser.add_argument("--output-dir", type=Path, default=ML_MODELS_DIR)
    parser.add_argument("--model", choices=["xgb", "rf", "lgbm"], default="xgb")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_split(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing feature file: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {"label", "class_name", "image_name"}
    if not required.issubset(df.columns):
        raise ValueError(f"{csv_path} must contain columns: {required}")
    if df.empty:
        raise ValueError(f"{csv_path} is empty")
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    feat_cols = [c for c in df.columns if c.startswith("feature_")]
    if not feat_cols:
        raise ValueError("No feature_* columns found")
    return feat_cols


def build_model(model_name: str, seed: int):
    if model_name == "xgb":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=450,
            max_depth=8,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=seed,
            n_jobs=-1,
        )

    if model_name == "lgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=450,
            learning_rate=0.06,
            num_leaves=63,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="multiclass",
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )

    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )


def evaluate_split(
    split_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_dir: Path,
) -> Dict:
    labels = list(range(len(class_names)))
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    report_csv = output_dir / f"classification_report_{split_name}.csv"
    pd.DataFrame(report_dict).T.to_csv(report_csv)

    cm_csv = output_dir / f"confusion_matrix_{split_name}.csv"
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(cm_csv)

    plt.figure(figsize=(12, 9))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix ({split_name})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    cm_png = output_dir / f"confusion_matrix_{split_name}.png"
    plt.savefig(cm_png, dpi=160)
    plt.close()

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "n_samples": int(len(y_true)),
        "report_csv": str(report_csv),
        "confusion_csv": str(cm_csv),
        "confusion_png": str(cm_png),
    }


def main() -> None:
    args = parse_args()
    features_dir = args.features_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_split(features_dir / "train_features.csv")
    valid_df = load_split(features_dir / "valid_features.csv")
    test_df = load_split(features_dir / "test_features.csv")

    feat_cols = get_feature_columns(train_df)
    for name, split_df in [("valid", valid_df), ("test", test_df)]:
        missing = [c for c in feat_cols if c not in split_df.columns]
        if missing:
            raise ValueError(f"{name} split missing feature columns; count={len(missing)}")

    # Class mapping is inferred from training split to keep labels stable.
    class_map = (
        train_df[["label", "class_name"]]
        .drop_duplicates()
        .sort_values("label")
        .set_index("label")["class_name"]
        .to_dict()
    )
    raw_train_labels = sorted(class_map.keys())
    class_names = [class_map[k] for k in raw_train_labels]

    # XGBoost requires labels to be contiguous [0..n_classes-1].
    raw_to_model = {raw: idx for idx, raw in enumerate(raw_train_labels)}
    model_to_raw = {idx: raw for raw, idx in raw_to_model.items()}

    def map_labels(raw_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mapped = np.array([raw_to_model.get(int(v), -1) for v in raw_labels], dtype=np.int64)
        known_mask = mapped >= 0
        return mapped, known_mask

    X_train = train_df[feat_cols].to_numpy(dtype=np.float32)
    y_train_raw = train_df["label"].to_numpy(dtype=np.int64)
    y_train, train_mask = map_labels(y_train_raw)
    if not np.all(train_mask):
        raise RuntimeError("Unexpected unmapped labels in training split")

    X_valid = valid_df[feat_cols].to_numpy(dtype=np.float32)
    y_valid_raw = valid_df["label"].to_numpy(dtype=np.int64)
    y_valid, valid_mask = map_labels(y_valid_raw)

    X_test = test_df[feat_cols].to_numpy(dtype=np.float32)
    y_test_raw = test_df["label"].to_numpy(dtype=np.int64)
    y_test, test_mask = map_labels(y_test_raw)

    model = build_model(args.model, args.seed)

    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    fit_kwargs = {"sample_weight": sample_weight}

    if args.model == "xgb":
        fit_kwargs["verbose"] = False

    model.fit(X_train, y_train, **fit_kwargs)

    model_joblib_path = output_dir / f"{args.model}_model.joblib"
    joblib.dump(model, model_joblib_path)

    xgb_native_path = None
    if args.model == "xgb":
        xgb_native_path = output_dir / "xgboost_model.json"
        model.save_model(str(xgb_native_path))

    y_pred_valid = model.predict(X_valid)
    y_pred_test = model.predict(X_test)

    metrics_valid = evaluate_split(
        "valid", y_valid[valid_mask], y_pred_valid[valid_mask], class_names, output_dir
    )
    metrics_test = evaluate_split(
        "test", y_test[test_mask], y_pred_test[test_mask], class_names, output_dir
    )

    metrics_valid["ignored_unseen_labels"] = int((~valid_mask).sum())
    metrics_test["ignored_unseen_labels"] = int((~test_mask).sum())

    pred_df = pd.DataFrame(
        {
            "image_name": test_df["image_name"],
            "true_label_raw": y_test_raw,
            "pred_label_raw": [model_to_raw[int(x)] for x in y_pred_test],
            "true_class": [class_map.get(int(x), str(x)) for x in y_test_raw],
            "pred_class": [class_map.get(model_to_raw[int(x)], str(int(x))) for x in y_pred_test],
            "is_seen_in_train": test_mask,
        }
    )
    pred_csv = output_dir / "test_predictions.csv"
    pred_df.to_csv(pred_csv, index=False)

    summary = {
        "model": args.model,
        "seed": args.seed,
        "features_dir": str(features_dir),
        "output_dir": str(output_dir),
        "n_features": len(feat_cols),
        "n_train": int(len(train_df)),
        "n_valid": int(len(valid_df)),
        "n_test": int(len(test_df)),
        "class_mapping": class_map,
        "label_mapping_raw_to_model": raw_to_model,
        "valid": metrics_valid,
        "test": metrics_test,
        "test_predictions_csv": str(pred_csv),
        "model_file": str(model_joblib_path),
        "xgb_native_model_file": str(xgb_native_path) if xgb_native_path else None,
    }

    out_json = output_dir / "results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 80)
    print("ML TRAINING COMPLETE")
    print("=" * 80)
    print(f"Model      : {args.model}")
    print(f"Model file : {model_joblib_path}")
    if xgb_native_path is not None:
        print(f"XGB native : {xgb_native_path}")
    print(f"Results    : {out_json}")
    print(f"Valid Acc  : {metrics_valid['accuracy']:.4f}")
    print(f"Valid F1-m : {metrics_valid['macro_f1']:.4f}")
    print(f"Test Acc   : {metrics_test['accuracy']:.4f}")
    print(f"Test F1-m  : {metrics_test['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
