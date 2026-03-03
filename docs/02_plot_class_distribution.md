# 📊 File 2: `plot_class_distribution.py` — Trực quan hóa phân phối lớp

> **Mục tiêu:** Hiểu tại sao cần xem phân phối dữ liệu trước khi training, và cách đọc biểu đồ.

---

## 📋 Code gốc (có chú thích từng dòng)

```python
# ─── IMPORT THƯ VIỆN ───
import os                    # Thao tác với file/folder
import pandas as pd          # Xử lý dữ liệu dạng bảng
import seaborn as sns        # Vẽ biểu đồ đẹp
import matplotlib.pyplot as plt  # Nền tảng vẽ biểu đồ

# ─── CẤU HÌNH ───
# Đường dẫn tới thư mục dataset gốc
root = r"C:\Users\ADMIN\Desktop\personal_train\ML_deppfeat\durian-1"

# Chỉ đếm trong 2 split: train và valid
splits = ["train", "valid"]

# Danh sách tên 15 lớp bệnh, theo đúng thứ tự class_id (0-14)
class_names = [
    'benh bo tri',           # ID 0
    'benh phan trang',       # ID 1
    'benh-chay-la',          # ID 2
    'benh-dom-mat-cua',      # ID 3
    'benh-gi-sat',           # ID 4
    'benh-la-vang',          # ID 5
    'benh-tao-do',           # ID 6
    'benh-than-thu',         # ID 7
    'dom la do nam',         # ID 8
    'dom sinh ly nhe',       # ID 9
    'la khoe binh thuong',   # ID 10
    'sau ve bua',            # ID 11
    'sau-an',                # ID 12
    'vang sinh ly',          # ID 13
    'vang thieu magie'       # ID 14
]

# ─── ĐẾM SỐ LƯỢNG TỪNG LỚP ───
records = []  # Danh sách lưu kết quả đếm

for split in splits:
    # Tìm đường dẫn tới thư mục labels
    # VD: durian-1/train/labels/
    label_dir = os.path.join(root, split, "labels")

    # Khởi tạo bộ đếm: 15 số 0 (mỗi số cho 1 lớp)
    counts = [0] * len(class_names)  # → [0, 0, 0, ..., 0]

    # Duyệt qua từng file nhãn trong thư mục labels
    for file in os.listdir(label_dir):
        path = os.path.join(label_dir, file)

        # Bỏ qua file rỗng (ảnh không có đối tượng nào)
        if os.path.getsize(path) == 0:
            continue

        # Đọc từng dòng trong file nhãn
        with open(path) as f:
            for line in f:
                # Lấy class_id (số đầu tiên)
                # VD: "2 0.45 0.62 0.15 0.20" → class_id = 2
                class_id = int(line.split()[0])

                # Tăng bộ đếm của lớp tương ứng
                counts[class_id] += 1

    # Lưu kết quả vào danh sách records
    for i, count in enumerate(counts):
        records.append({
            "Split": split,          # "train" hoặc "valid"
            "Class": class_names[i], # Tên lớp bệnh
            "Count": count           # Số lượng instances
        })

# Tạo DataFrame (bảng dữ liệu) từ records
df = pd.DataFrame(records)

# ─── VẼ BIỂU ĐỒ ───
plt.figure(figsize=(14, 6))  # Kích thước: 14 inch rộng × 6 inch cao

# Vẽ biểu đồ cột nhóm (grouped bar chart)
# x = tên lớp, y = số lượng, hue = split (train/valid = 2 màu khác nhau)
sns.barplot(data=df, x="Class", y="Count", hue="Split")

plt.xticks(rotation=75)      # Xoay nhãn trục X 75° để đọc được
plt.title("Class Distribution - Train vs Valid")
plt.tight_layout()            # Tự điều chỉnh layout không bị cắt
plt.show()                    # Hiển thị biểu đồ
```

---

## 🧩 Giải thích từng khái niệm

### Tại sao cần xem phân phối dữ liệu?

Đây là bước **CỰC KỲ QUAN TRỌNG** mà nhiều người mới bỏ qua.

```
Tưởng tượng bạn dạy 1 đứa bé phân biệt 3 loại trái cây:

  Trường hợp 1 (Cân bằng):
  - Cho xem 100 ảnh táo ✅
  - Cho xem 100 ảnh cam ✅  → Bé học tốt cả 3 loại
  - Cho xem 100 ảnh chuối ✅

  Trường hợp 2 (Mất cân bằng): ← Dự án này gặp phải!
  - Cho xem 1000 ảnh táo 😱
  - Cho xem 1000 ảnh cam 😱   → Bé chỉ nhớ táo và cam
  - Cho xem 5 ảnh chuối 😢      → "Chuối là gì?" 🤷
```

### Pandas là gì?

**Pandas** = thư viện xử lý dữ liệu dạng bảng (giống Excel trong Python).

```python
# DataFrame giống 1 bảng Excel:
#   Split  |  Class        |  Count
#   train  |  benh bo tri  |  0
#   train  |  benh-chay-la |  759
#   valid  |  benh-chay-la |  150
#   ...
```

### Seaborn + Matplotlib là gì?

```
Matplotlib = bộ vẽ cơ bản (giống bút chì)
Seaborn    = bộ vẽ cao cấp (giống bút màu + template đẹp)

Seaborn được xây dựng trên Matplotlib, nên cần cả hai.
```

### Cách đọc biểu đồ kết quả

```
  1600 │ ██
       │ ██                              ██
  1200 │ ██                              ██
       │ ██     ██                       ██
   800 │ ██     ██                       ██
       │ ██     ██          ██           ██
   400 │ ██     ██    ██    ██     ██    ██
       │ ██     ██    ██    ██     ██    ██  ▒▒
     0 │─▒▒─────██────██────██──▒▒─██──▒▒██──▒▒──▒▒──▒▒──▒▒──▒▒
       benh  benh   benh  benh  benh  benh  la  sau sau vang vang
       botri  phan  chay  dom   gisat than  khoe ve  an  sinh thieu
             trang   la   mat         thu       bua     ly  magie

  ██ = train    ▒▒ = valid (validation)

  VẤN ĐỀ: benh phan trang (1653) và sau-an (1338) quá nhiều!
           benh bo tri, benh-la-vang, vang sinh ly... gần bằng 0!
```

---

## 💡 Bài học rút ra

| Quan sát                     | Ý nghĩa              | Cần làm gì                     |
| ---------------------------- | -------------------- | ------------------------------ |
| Lớp "benh phan trang" = 1653 | Quá nhiều, chiếm 32% | Cần **giảm bớt** (downsample)  |
| Lớp "sau-an" = 1338          | Quá nhiều, chiếm 26% | Cần **giảm bớt**               |
| Lớp "benh bo tri" = 0        | Không có mẫu!        | **Bỏ luôn** hoặc thu thập thêm |
| 8 lớp có < 32 mẫu            | Quá ít, không đủ học | **Gộp chung** thành 1 lớp      |

> **Kết luận:** Dữ liệu **mất cân bằng nghiêm trọng** → cần xử lý ở bước tiếp theo (refactor + downsample).

---

## 📝 Giải thích cú pháp Python

### `os.path.join(root, split, "labels")`

```python
# Nối đường dẫn theo đúng format của hệ điều hành

# Windows:
os.path.join("C:\\data", "train", "labels")
# → "C:\\data\\train\\labels"

# Mac/Linux:
os.path.join("/data", "train", "labels")
# → "/data/train/labels"
```

### `line.split()[0]`

```python
# split() = tách chuỗi thành list theo dấu cách
# [0] = lấy phần tử đầu tiên

line = "2 0.45 0.62 0.15 0.20"
line.split()     # → ["2", "0.45", "0.62", "0.15", "0.20"]
line.split()[0]  # → "2"
int("2")         # → 2 (chuyển từ text sang số)
```

### `enumerate(counts)`

```python
# enumerate = duyệt danh sách kèm theo vị trí (index)

counts = [100, 200, 50]
for i, count in enumerate(counts):
    print(f"Vị trí {i}: giá trị {count}")
# → Vị trí 0: giá trị 100
# → Vị trí 1: giá trị 200
# → Vị trí 2: giá trị 50
```

---

> **Đọc tiếp:** [03_refactor_data.md](./03_refactor_data.md)
