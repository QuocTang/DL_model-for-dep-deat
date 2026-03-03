import os
import random
import shutil

# ====== CONFIG ======
root = r"C:\Users\ADMIN\Desktop\personal_train\ML_deppfeat\durian-1"
split = "train"

target_classes = [2, 12]  
# 2 = benh-chay-la
# 12 = sau-an

keep_ratio = 0.4   # giữ lại 40%

# ====================

image_dir = os.path.join(root, split, "images")
label_dir = os.path.join(root, split, "labels")

all_images = os.listdir(image_dir)

large_class_images = []

for img_name in all_images:
    label_path = os.path.join(label_dir, img_name.replace(".jpg", ".txt"))

    if not os.path.exists(label_path):
        continue

    with open(label_path) as f:
        lines = f.readlines()

    for line in lines:
        class_id = int(line.split()[0])
        if class_id in target_classes:
            large_class_images.append(img_name)
            break

# unique
large_class_images = list(set(large_class_images))

print("Images containing large classes:", len(large_class_images))

# chọn ảnh giữ lại
keep_count = int(len(large_class_images) * keep_ratio)
keep_images = set(random.sample(large_class_images, keep_count))

print("Keeping:", keep_count)

# Tạo folder mới
new_root = root + "_balanced"
new_image_dir = os.path.join(new_root, split, "images")
new_label_dir = os.path.join(new_root, split, "labels")

os.makedirs(new_image_dir, exist_ok=True)
os.makedirs(new_label_dir, exist_ok=True)

for img_name in all_images:

    label_path = os.path.join(label_dir, img_name.replace(".jpg", ".txt"))

    if img_name in large_class_images:
        if img_name not in keep_images:
            continue

    shutil.copy(
        os.path.join(image_dir, img_name),
        os.path.join(new_image_dir, img_name)
    )

    shutil.copy(
        label_path,
        os.path.join(new_label_dir, img_name.replace(".jpg", ".txt"))
    )

print("Done. New dataset created.")