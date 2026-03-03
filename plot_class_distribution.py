import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ====== ROOT DATASET (folder chứa train/ valid/ test) ======
root = r"C:\Users\ADMIN\Desktop\personal_train\ML_deppfeat\durian-1"

splits = ["train", "valid"]

class_names = [
    'benh bo tri', 'benh phan trang', 'benh-chay-la', 'benh-dom-mat-cua',
    'benh-gi-sat', 'benh-la-vang', 'benh-tao-do', 'benh-than-thu',
    'dom la do nam', 'dom sinh ly nhe', 'la khoe binh thuong',
    'sau ve bua', 'sau-an', 'vang sinh ly', 'vang thieu magie'
]

records = []

for split in splits:
    label_dir = os.path.join(root, split, "labels")
    counts = [0] * len(class_names)

    for file in os.listdir(label_dir):
        path = os.path.join(label_dir, file)

        if os.path.getsize(path) == 0:
            continue

        with open(path) as f:
            for line in f:
                class_id = int(line.split()[0])
                counts[class_id] += 1

    for i, count in enumerate(counts):
        records.append({
            "Split": split,
            "Class": class_names[i],
            "Count": count
        })

df = pd.DataFrame(records)

# ====== VẼ ======
plt.figure(figsize=(14,6))
sns.barplot(data=df, x="Class", y="Count", hue="Split")
plt.xticks(rotation=75)
plt.title("Class Distribution - Train vs Valid")
plt.tight_layout()
plt.show()