import os
import shutil
import sys
from pathlib import Path

# Thêm thư mục cha vào sys.path để import config``
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import SRC_ROOT, DST_ROOT

# Mapping of old class IDs to new sequential IDs (có thể chỉnh sửa)
KEEP_IDS = {
    12: 0,  # sau-an
    2: 1,   # benh-chay-la
    7: 2,   # benh-than-thu
    3: 3,   # benh-dom-mat-cua
    10: 4,  # la khoe binh thuong
}

def new_class_id(old_id: int) -> int:
    """Return the mapped class ID, or fallback to 5 for unspecified classes."""
    return KEEP_IDS.get(old_id, 5)

for split in ["train", "valid", "test"]:
    # Đảm bảo thư mục đích tồn tại
    (DST_ROOT / split / "images").mkdir(parents=True, exist_ok=True)
    (DST_ROOT / split / "labels").mkdir(parents=True, exist_ok=True)

    src_img_dir = SRC_ROOT / split / "images"
    src_lbl_dir = SRC_ROOT / split / "labels"
    dst_img_dir = DST_ROOT / split / "images"
    dst_lbl_dir = DST_ROOT / split / "labels"

    # Sao chép ảnh
    for file_name in os.listdir(src_img_dir):
        shutil.copy(src_img_dir / file_name, dst_img_dir / file_name)

    # Chuyển đổi và ghi lại nhãn
    for file_name in os.listdir(src_lbl_dir):
        src_file = src_lbl_dir / file_name
        dst_file = dst_lbl_dir / file_name
        with open(src_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            # Hỗ trợ cả bbox (5 token) lẫn polygon (>5 token, lẻ).
            if len(parts) < 5:
                continue
            old_id = int(parts[0])
            parts[0] = str(new_class_id(old_id))
            new_lines.append(" ".join(parts))
        with open(dst_file, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines))

print("DONE SAFE REFACTOR ✅")