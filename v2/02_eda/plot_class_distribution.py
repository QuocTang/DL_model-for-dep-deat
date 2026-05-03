"""EDA: vẽ biểu đồ phân bố nhãn theo split (train / valid / test).

Đọc class_names từ data.yaml để không hard-code.
"""

import os
import sys
from pathlib import Path

import matplotlib

# Dùng backend Agg nếu chạy headless (WSL không có DISPLAY).
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_YAML, DST_ROOT, EDA_DIR

with open(DATA_YAML, "r", encoding="utf-8") as f:
    data_cfg = yaml.safe_load(f)

class_names = data_cfg["names"]
nc = data_cfg["nc"]

root = str(DST_ROOT)
splits = ["train", "valid", "test"]

records = []
for split in splits:
    label_dir = os.path.join(root, split, "labels")
    if not os.path.isdir(label_dir):
        print(f"[WARN] Bỏ qua split '{split}' (không tồn tại): {label_dir}")
        continue

    counts = [0] * nc
    for file in os.listdir(label_dir):
        path = os.path.join(label_dir, file)
        if os.path.getsize(path) == 0:
            continue
        with open(path) as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                class_id = int(parts[0])
                if 0 <= class_id < nc:
                    counts[class_id] += 1

    for i, count in enumerate(counts):
        records.append({"Split": split, "Class": class_names[i], "Count": count})

df = pd.DataFrame(records)

print("\n=== Phân bố nhãn ===")
print(df.pivot(index="Class", columns="Split", values="Count").fillna(0).astype(int))

plt.figure(figsize=(12, 6))
sns.barplot(data=df, x="Class", y="Count", hue="Split")
plt.xticks(rotation=30, ha="right")
plt.title(f"Class Distribution ({nc} classes) — train / valid / test")
plt.tight_layout()

EDA_DIR.mkdir(parents=True, exist_ok=True)
out_path = EDA_DIR / "class_distribution.png"
csv_path = EDA_DIR / "class_distribution.csv"

df.pivot(index="Class", columns="Split", values="Count").fillna(0).astype(int).to_csv(csv_path)
plt.savefig(out_path, dpi=120)
print(f"\nLưu biểu đồ : {out_path}")
print(f"Lưu bảng    : {csv_path}")

if os.environ.get("DISPLAY"):
    plt.show()
