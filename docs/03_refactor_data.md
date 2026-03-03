# 🔄 File 3: `refactor_data.py` — Tái cấu trúc nhãn dữ liệu

> **Mục tiêu:** Hiểu tại sao cần gộp lớp, cách ánh xạ (mapping) class ID, và quy trình xử lý file nhãn.

---

## 📋 Code gốc (có chú thích từng dòng)

```python
import os
import shutil    # Thư viện copy/move file

# ─── ĐƯỜNG DẪN ───
# Dữ liệu gốc (15 lớp)
src_root = r"C:\Users\ADMIN\Desktop\personal_train\ML_deppfeat\durian-1"
# Dữ liệu sau refactor (6 lớp)
dst_root = r"C:\Users\ADMIN\Desktop\personal_train\ML_deppfeat\durian_refactor"

# ─── BẢNG ÁNH XẠ ───
# Dictionary: {class_id_cũ: class_id_mới}
# Chỉ giữ 5 lớp có đủ dữ liệu, đánh số lại từ 0-4
keep_ids = {
    12: 0,  # sau-an           → class mới 0
    2: 1,   # benh-chay-la     → class mới 1
    7: 2,   # benh-than-thu    → class mới 2
    3: 3,   # benh-dom-mat-cua → class mới 3
    10: 4   # la khoe          → class mới 4
}

# Hàm chuyển đổi class ID
def new_class_id(old_id):
    # Nếu old_id nằm trong keep_ids → trả về ID mới
    # Nếu không → trả về 5 (lớp "other")
    return keep_ids.get(old_id, 5)

# ─── XỬ LÝ TỪNG SPLIT ───
for split in ["train", "valid", "test"]:
    # Tạo thư mục đích (nếu chưa có)
    os.makedirs(os.path.join(dst_root, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(dst_root, split, "labels"), exist_ok=True)

    # Đường dẫn nguồn
    img_src = os.path.join(src_root, split, "images")
    lbl_src = os.path.join(src_root, split, "labels")

    # Đường dẫn đích
    img_dst = os.path.join(dst_root, split, "images")
    lbl_dst = os.path.join(dst_root, split, "labels")

    # BƯỚC 1: Copy ảnh nguyên vẹn (không thay đổi gì)
    for file in os.listdir(img_src):
        shutil.copy(os.path.join(img_src, file), img_dst)

    # BƯỚC 2: Đọc file nhãn, đổi class ID, ghi file nhãn mới
    for file in os.listdir(lbl_src):
        src_file = os.path.join(lbl_src, file)
        dst_file = os.path.join(lbl_dst, file)

        # Đọc tất cả dòng trong file nhãn
        with open(src_file, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()

            # Bỏ qua dòng không đúng format (phải có đúng 5 phần tử)
            if len(parts) != 5:
                continue

            # Lấy class ID cũ, chuyển sang class ID mới
            old_id = int(parts[0])
            parts[0] = str(new_class_id(old_id))

            # Ghép lại thành 1 dòng
            new_lines.append(" ".join(parts))

        # Ghi file nhãn mới
        with open(dst_file, "w") as f:
            f.write("\n".join(new_lines))

print("DONE SAFE REFACTOR ✅")
```

---

## 🧩 Giải thích chi tiết

### Tại sao cần gộp lớp?

Từ biểu đồ ở bước trước, ta thấy:

```
15 LỚP GỐC:
                                    Số mẫu
  benh phan trang    ████████████████ 1653  ← Quá nhiều
  sau-an             █████████████    1338  ← Quá nhiều
  benh-chay-la       ███████          759
  benh-than-thu      █████            563
  benh-dom-mat-cua   ██               246
  la khoe            ██               228
  benh-gi-sat        █                130
  benh bo tri        ▏                 0   ← Không có!
  benh la vang       ▏                 5
  benh-tao-do        ▏                32
  dom la do nam      ▏                28
  dom sinh ly nhe    ▏                 6
  sau ve bua         ▏                 6
  vang sinh ly       ▏                14
  vang thieu magie   ▏                32

VẤN ĐỀ: 8 lớp có < 32 mẫu → Model KHÔNG THỂ HỌC được!
```

**Giải pháp:** Giữ 5 lớp có đủ dữ liệu, gộp phần còn lại:

```
SAU REFACTOR (6 LỚP):

  ID mới 0: sau-an            (từ ID cũ 12)   1338 mẫu ✅
  ID mới 1: benh-chay-la      (từ ID cũ 2)     759 mẫu ✅
  ID mới 2: benh-than-thu     (từ ID cũ 7)     563 mẫu ✅
  ID mới 3: benh-dom-mat-cua  (từ ID cũ 3)     246 mẫu ✅
  ID mới 4: la-khoe           (từ ID cũ 10)    228 mẫu ✅
  ID mới 5: other             (tất cả còn lại)  ~253 mẫu
```

### Dictionary `.get()` hoạt động thế nào?

```python
keep_ids = {12: 0, 2: 1, 7: 2, 3: 3, 10: 4}

# .get(key, default) = tìm key, nếu không có → trả default

keep_ids.get(12, 5)   # → 0  (tìm thấy 12 → trả về 0)
keep_ids.get(2, 5)    # → 1  (tìm thấy 2 → trả về 1)
keep_ids.get(99, 5)   # → 5  (không tìm thấy 99 → trả về 5)
keep_ids.get(0, 5)    # → 5  (không tìm thấy 0 → trả về 5)
```

### Quy trình xử lý 1 file nhãn

```
FILE GỐC (img001.txt):              FILE MỚI (img001.txt):
┌─────────────────────────┐         ┌─────────────────────────┐
│ 2 0.45 0.62 0.15 0.20   │  →→→   │ 1 0.45 0.62 0.15 0.20   │
│ 12 0.78 0.31 0.10 0.08  │  →→→   │ 0 0.78 0.31 0.10 0.08   │
│ 0 0.55 0.40 0.12 0.15   │  →→→   │ 5 0.55 0.40 0.12 0.15   │
└─────────────────────────┘         └─────────────────────────┘

Giải thích:
  Dòng 1: class 2 (benh-chay-la) → class 1 (mapping: 2→1)
  Dòng 2: class 12 (sau-an)      → class 0 (mapping: 12→0)
  Dòng 3: class 0 (benh bo tri)  → class 5 (không trong mapping → default 5)

  Tọa độ (4 số sau) giữ nguyên không đổi!
```

### `shutil.copy()` — Copy file

```python
import shutil

# Copy 1 file từ nguồn → đích
shutil.copy("nguồn/img001.jpg", "đích/img001.jpg")

# Tương đương với việc bạn copy-paste file trong File Explorer
```

### `os.makedirs(path, exist_ok=True)`

```python
# Tạo thư mục (kể cả thư mục cha nếu chưa có)

os.makedirs("data/train/images", exist_ok=True)
# → Tạo: data/ → data/train/ → data/train/images/

# exist_ok=True: Nếu thư mục đã tồn tại → KHÔNG báo lỗi
# exist_ok=False (mặc định): Nếu đã tồn tại → BÁO LỖI
```

### `with open(file, "r") as f:` — Đọc/Ghi file

```python
# ĐỌC file (mode "r" = read):
with open("img001.txt", "r") as f:
    lines = f.readlines()   # Đọc tất cả dòng → list
    # lines = ["2 0.45 0.62 0.15 0.20\n", "12 0.78 ..."]

# GHI file (mode "w" = write):
with open("img001.txt", "w") as f:
    f.write("nội dung mới")   # Ghi đè toàn bộ file

# "with" = tự đóng file sau khi xong (best practice!)
```

---

## 💡 Bài học rút ra

### 1. Tại sao đánh số lại từ 0?

YOLO yêu cầu class ID phải **liên tục từ 0**:

- ❌ Sai: ID = [2, 3, 7, 10, 12] (không liên tục, thiếu 0, 1, 4...)
- ✅ Đúng: ID = [0, 1, 2, 3, 4, 5] (liên tục từ 0)

### 2. Tại sao gộp thành "other" thay vì xóa?

Nếu xóa hoàn toàn nhãn các lớp ít mẫu:

- Ảnh vẫn có vùng bệnh đó hiển thị
- Model sẽ bối rối vì thấy nhưng không có nhãn
- Gộp thành "other" → model biết "có cái gì đó ở đây" nhưng không cần phân loại chi tiết

### 3. Pattern: Xử lý dữ liệu không destructive

```
✅ ĐÚNG (dự án này làm):
   Dữ liệu gốc (durian-1/) → Copy sang → Dữ liệu mới (durian_refactor/)
   → Giữ nguyên bản gốc, nếu sai có thể làm lại

❌ SAI:
   Sửa trực tiếp dữ liệu gốc
   → Nếu sai, mất dữ liệu vĩnh viễn!
```

---

> **Đọc tiếp:** [04_downsample_large_classes.md](./04_downsample_large_classes.md)
