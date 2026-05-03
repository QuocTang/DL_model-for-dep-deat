"""
Select top classes by validation performance and data support, then export filtered feature CSVs.

Default strategy:
- Rank classes by validation F1-score (desc), tie-break by train support (desc)
- Ignore classes with train support < min_train_samples
- Keep top_k classes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter feature CSVs to top classes")
    parser.add_argument("--input-dir", type=Path, default=Path("extracted_features"))
    parser.add_argument("--report-csv", type=Path, default=Path("ml_models/classification_report_valid.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("extracted_features_top6"))
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--min-train-samples", type=int, default=30)
    parser.add_argument(
        "--class-list",
        type=str,
        default=None,
        help="Comma-separated class names to keep (manual mode, skip auto ranking)",
    )
    return parser.parse_args()


def load_split(input_dir: Path, split_name: str) -> pd.DataFrame:
    csv_path = input_dir / f"{split_name}_features.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing split file: {csv_path}")
    return pd.read_csv(csv_path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_split(args.input_dir, "train")
    valid_df = load_split(args.input_dir, "valid")
    test_df = load_split(args.input_dir, "test")

    train_counts = train_df["class_name"].value_counts()

    if args.class_list:
        selected_classes = [x.strip() for x in args.class_list.split(",") if x.strip()]
        selected = []
        for class_name in selected_classes:
            selected.append(
                {
                    "class_name": class_name,
                    "train_count": int(train_counts.get(class_name, 0)),
                    "valid_f1": None,
                }
            )
    else:
        report_df = pd.read_csv(args.report_csv, index_col=0)
        scored_rows = []
        for class_name, count in train_counts.items():
            if int(count) < args.min_train_samples:
                continue
            f1 = float(report_df.loc[class_name, "f1-score"]) if class_name in report_df.index else -1.0
            scored_rows.append({"class_name": class_name, "train_count": int(count), "valid_f1": f1})

        if not scored_rows:
            raise RuntimeError("No classes satisfy min_train_samples")

        scored_rows.sort(key=lambda x: (x["valid_f1"], x["train_count"]), reverse=True)
        selected = scored_rows[: args.top_k]
        selected_classes = [x["class_name"] for x in selected]

    def filter_split(df: pd.DataFrame) -> pd.DataFrame:
        out = df[df["class_name"].isin(selected_classes)].copy()
        return out

    train_f = filter_split(train_df)
    valid_f = filter_split(valid_df)
    test_f = filter_split(test_df)

    # Keep raw labels for traceability, but remap to contiguous labels for cleaner training.
    old_to_new = {
        int(old_label): idx
        for idx, old_label in enumerate(sorted(train_f[["label", "class_name"]]["label"].unique().tolist()))
    }

    for df in (train_f, valid_f, test_f):
        df["label_raw"] = df["label"].astype(int)
        df["label"] = df["label_raw"].map(old_to_new)

    train_f.to_csv(args.output_dir / "train_features.csv", index=False)
    valid_f.to_csv(args.output_dir / "valid_features.csv", index=False)
    test_f.to_csv(args.output_dir / "test_features.csv", index=False)

    summary = {
        "input_dir": str(args.input_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "top_k": args.top_k,
        "min_train_samples": args.min_train_samples,
        "class_list": args.class_list,
        "selected_classes": selected,
        "label_old_to_new": old_to_new,
        "n_train": int(len(train_f)),
        "n_valid": int(len(valid_f)),
        "n_test": int(len(test_f)),
    }

    with open(args.output_dir / "selection_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Selected classes:")
    for row in selected:
        print(f"- {row['class_name']}: train={row['train_count']}, valid_f1={row['valid_f1']:.4f}")
    print(f"Saved filtered CSVs to: {args.output_dir}")


if __name__ == "__main__":
    main()
