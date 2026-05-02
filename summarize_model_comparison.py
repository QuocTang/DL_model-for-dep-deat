"""
Aggregate multiple model result folders into comparison CSV/Markdown reports.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize model comparison folders")
    parser.add_argument("--comparison-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []

    for result_path in sorted(args.comparison_dir.glob("*/results.json")):
        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        rows.append(
            {
                "model": data["model"],
                "n_train": data["n_train"],
                "n_valid": data["n_valid"],
                "n_test": data["n_test"],
                "valid_accuracy": data["valid"]["accuracy"],
                "valid_macro_f1": data["valid"]["macro_f1"],
                "valid_weighted_f1": data["valid"]["weighted_f1"],
                "test_accuracy": data["test"]["accuracy"],
                "test_macro_f1": data["test"]["macro_f1"],
                "test_weighted_f1": data["test"]["weighted_f1"],
                "results_json": str(result_path.resolve()),
                "model_file": data.get("model_file"),
            }
        )

    if not rows:
        raise RuntimeError(f"No results.json found under {args.comparison_dir}")

    df = pd.DataFrame(rows).sort_values(["test_macro_f1", "test_accuracy"], ascending=False)
    csv_path = args.comparison_dir / "model_comparison_summary.csv"
    md_path = args.comparison_dir / "model_comparison_summary.md"
    df.to_csv(csv_path, index=False)

    md_lines = [
        "# Model Comparison Summary",
        "",
        "| Model | Valid Acc | Valid Macro-F1 | Test Acc | Test Macro-F1 | Model File |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for _, row in df.iterrows():
        md_lines.append(
            f"| {row['model']} | {row['valid_accuracy']:.4f} | {row['valid_macro_f1']:.4f} | {row['test_accuracy']:.4f} | {row['test_macro_f1']:.4f} | {row['model_file']} |"
        )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"Saved: {csv_path}")
    print(f"Saved: {md_path}")
    print(df[["model", "valid_accuracy", "valid_macro_f1", "test_accuracy", "test_macro_f1"]])


if __name__ == "__main__":
    main()
