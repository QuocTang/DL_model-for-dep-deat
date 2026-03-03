# ⚖️ File 4: `downsample_large_classes.py` — Cân bằng dữ liệu

> **Mục tiêu:** Hiểu vấn đề "dữ liệu mất cân bằng" (imbalanced data), cách Downsampling hoạt động, và kỹ thuật random sampling.

---

## 📋 Code gốc (có chú thích từng dòng)

```python
import os
import random     # Thư viện tạo số ngẫu nhiên
import shutil     # Copy file

# ─── CẤU HÌNH ───
root = r"C:\Users\ADMIN\Desktop\personal_train\ML_deppfeat\durian-1"
split = "train"       # Chỉ downsample tập train

target_classes = [2, 12]   # Class IDs cần giảm
# 2 = benh-chay-la (759 mẫu)
# 12 = sau-an (1338 mẫu)

keep_ratio = 0.4   # Giữ lại 40%, bỏ 60%

# ─── ĐƯỜNG DẪN ───
image_dir = os.path.join(root, split, "images")
label_dir = os.path.join(root, split, "labels")

# Lấy danh sách tất cả file ảnh
all_images = os.listdir(image_dir)

# ─── BƯỚC 1: TÌM ẢNH CHỨA LỚP LỚN ───
large_class_images = []

for img_name in all_images:
    # Tìm file nhãn tương ứng (đổi .jpg → .txt)
    # VD: "img001.jpg" → "img001.txt"
    label_path = os.path.join(label_dir, img_name.replace(".jpg", ".txt"))

    # Bỏ qua nếu không có file nhãn
    if not os.path.exists(label_path):
        continue

    # Đọc file nhãn
    with open(label_path) as f:
        lines = f.readlines()

    # Kiểm tra: ảnh này có chứa target class không?
    for line in lines:
        class_id = int(line.split()[0])
        if class_id in target_classes:
            large_class_images.append(img_name)
            break   # Tìm thấy rồi → không cần đọc tiếp các dòng khác

# Loại bỏ trùng lặp
large_class_images = list(set(large_class_images))

print("Images containing large classes:", len(large_class_images))

# ─── BƯỚC 2: RANDOM CHỌN ẢNH GIỮ LẠI ───
# Giữ 40% ảnh
keep_count = int(len(large_class_images) * keep_ratio)
# random.sample = chọn ngẫu nhiên keep_count ảnh từ danh sách
keep_images = set(random.sample(large_class_images, keep_count))

print("Keeping:", keep_count)

# ─── BƯỚC 3: COPY ẢNH + NHÃN SANG FOLDER MỚI ───
new_root = root + "_balanced"
new_image_dir = os.path.join(new_root, split, "images")
new_label_dir = os.path.join(new_root, split, "labels")

os.makedirs(new_image_dir, exist_ok=True)
os.makedirs(new_label_dir, exist_ok=True)

for img_name in all_images:
    label_path = os.path.join(label_dir, img_name.replace(".jpg", ".txt"))

    # NẾU ảnh thuộc lớp lớn VÀ KHÔNG được chọn giữ → BỎ QUA
    if img_name in large_class_images:
        if img_name not in keep_images:
            continue    # ← skip! không copy ảnh này

    # Copy ảnh + nhãn sang folder mới
    shutil.copy(
        os.path.join(image_dir, img_name),
        os.path.join(new_image_dir, img_name)
    )

    shutil.copy(
        label_path,
        os.path.join(new_label_dir, img_name.replace(".jpg", ".txt"))
    )

print("Done. New dataset created.")
```

---

## 🧩 Giải thích chi tiết

### Vấn đề: Imbalanced Data (dữ liệu mất cân bằng)

```
TRƯỚC KHI CÂN BẰNG:

  benh-chay-la    ████████████████████████████████████████  759
  sau-an          ████████████████████████████████████████████████████████  1338
  benh-than-thu   ██████████████████████████████  563
  benh-dom-mat-cua █████████████  246
  la-khoe          ████████████  228

  → Model sẽ "thiên vị" (bias) lớp nhiều mẫu
  → Khi gặp ảnh lạ, nó sẽ luôn đoán "sau-an" vì đó là lớp nó thấy nhiều nhất
```

### Giải pháp: Downsampling

```
Downsampling = GIẢM SỐ LƯỢNG ảnh của lớp chiếm đa số

  keep_ratio = 0.4 (giữ 40%)

  VÍ DỤ với 1000 ảnh chứa lớp lớn:
  ├── 400 ảnh được random CHỌN → ✅ GIỮ LẠI (copy sang folder mới)
  └── 600 ảnh không được chọn  → ❌ BỎ QUA (không copy)

  Ảnh KHÔNG chứa lớp lớn → ✅ GIỮ LẠI 100% (luôn copy)
```

### Flow chart chi tiết

```
    Duyệt từng ảnh trong all_images
              │
              ▼
    ┌─────────────────────────┐
    │ Ảnh này chứa lớp lớn?   │
    │ (class 2 hoặc 12)       │
    └──────┬──────────┬───────┘
           │          │
          CÓ        KHÔNG
           │          │
           ▼          ▼
    ┌────────────┐  ┌──────────────┐
    │ Ảnh được   │  │ COPY ngay!   │
    │ chọn giữ?  │  │ (giữ 100%)  │
    └──┬─────┬───┘  └──────────────┘
       │     │
      CÓ   KHÔNG
       │     │
       ▼     ▼
    COPY!   BỎ QUA
```

### `random.sample()` — Chọn ngẫu nhiên

```python
import random

danh_sach = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

# Chọn 4 phần tử ngẫu nhiên (không trùng lặp)
ket_qua = random.sample(danh_sach, 4)
# Ví dụ: → ["C", "F", "A", "I"]

# Mỗi lần chạy → kết quả KHÁC NHAU (vì random)
# → Đó là lý do kết quả training mỗi lần có thể khác nhau
```

### `set()` — Tập hợp (loại bỏ trùng lặp)

```python
# list có thể trùng lặp:
my_list = ["img1.jpg", "img1.jpg", "img2.jpg"]  # img1 xuất hiện 2 lần

# set tự loại bỏ trùng:
my_set = set(my_list)  # → {"img1.jpg", "img2.jpg"}

# Tại sao cần set ở đây?
# Vì 1 ảnh có thể chứa CẢ class 2 VÀ class 12
# → append 2 lần → cần loại bỏ trùng
```

### `break` — Dừng vòng lặp sớm

```python
for line in lines:
    class_id = int(line.split()[0])
    if class_id in target_classes:
        large_class_images.append(img_name)
        break   # ← DỪNG! Không cần đọc tiếp

# Tại sao break?
# VD file nhãn có 5 dòng, dòng 1 đã là class 2 → đủ rồi!
# Không cần kiểm tra 4 dòng còn lại → tiết kiệm thời gian
```

### `in` — Kiểm tra phần tử thuộc tập hợp

```python
target_classes = [2, 12]

5 in target_classes    # → False (5 không nằm trong [2, 12])
2 in target_classes    # → True  (2 nằm trong [2, 12])
12 in target_classes   # → True

# Với set, phép "in" NHANH HƠN nhiều so với list
# set: O(1) — tức thì
# list: O(n) — phải duyệt từng phần tử
```

---

## 💡 Bài học rút ra

### 1. Downsampling vs Upsampling

```
Có 2 cách cân bằng dữ liệu:

  Downsampling (dự án này dùng):
    Giảm lớp nhiều → ngang bằng lớp ít
    + Đơn giản, nhanh
    - Mất dữ liệu (bỏ 60% ảnh!)

  Upsampling:
    Tăng lớp ít → ngang bằng lớp nhiều
    + Không mất dữ liệu
    - Dễ overfitting (model thấy ảnh giống nhau lặp lại)

  Augmentation (tốt nhất):
    Tăng lớp ít bằng BIẾN ĐỔI ẢNH (xoay, lật, đổi màu...)
    + Tạo ảnh "mới" từ ảnh cũ
    + Model học đa dạng hơn
    - Phức tạp hơn
```

### 2. Hạn chế của code này

```
⚠️ Chỉ downsample tập train:
   Code chỉ xử lý split = "train"
   → valid và test giữ nguyên (đúng, vì ta muốn đánh giá trên dữ liệu thật)

⚠️ Không cố định random seed:
   Mỗi lần chạy → chọn ảnh khác nhau
   → Kết quả không reproducible (không tái tạo được)

   Cách fix: thêm random.seed(42) ở đầu file
```

---

> **Đọc tiếp:** [05_model_pretrain.md](./05_model_pretrain.md)
