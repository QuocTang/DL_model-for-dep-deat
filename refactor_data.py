import os
import shutil

# ====== ĐƯỜNG DẪN ======
src_root = r"C:\Users\ADMIN\Desktop\personal_train\ML_deppfeat\durian-1"
dst_root = r"C:\Users\ADMIN\Desktop\personal_train\ML_deppfeat\durian_refactor"

# ====== MAPPING CHUẨN ======
keep_ids = {
    12: 0,  # sau-an
    2: 1,   # benh-chay-la
    7: 2,   # benh-than-thu
    3: 3,   # benh-dom-mat-cua
    10: 4   # la khoe binh thuong
}

def new_class_id(old_id):
    return keep_ids.get(old_id, 5)

for split in ["train", "valid", "test"]:
    os.makedirs(os.path.join(dst_root, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(dst_root, split, "labels"), exist_ok=True)

    img_src = os.path.join(src_root, split, "images")
    lbl_src = os.path.join(src_root, split, "labels")

    img_dst = os.path.join(dst_root, split, "images")
    lbl_dst = os.path.join(dst_root, split, "labels")

    # copy images
    for file in os.listdir(img_src):
        shutil.copy(os.path.join(img_src, file), img_dst)

    # rewrite labels
    for file in os.listdir(lbl_src):
        src_file = os.path.join(lbl_src, file)
        dst_file = os.path.join(lbl_dst, file)

        with open(src_file, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            old_id = int(parts[0])
            parts[0] = str(new_class_id(old_id))
            new_lines.append(" ".join(parts))

        with open(dst_file, "w") as f:
            f.write("\n".join(new_lines))

print("DONE SAFE REFACTOR ✅")