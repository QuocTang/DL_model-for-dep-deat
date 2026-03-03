# 📥 File 1: `pull_data.py` — Tải dữ liệu từ Roboflow

> **Mục tiêu:** Hiểu cách tải dataset (tập dữ liệu) ảnh đã gán nhãn từ nền tảng Roboflow về máy tính.

---

## 📋 Code gốc (có chú thích)

```python
# Dòng 1: Import thư viện Roboflow
# "from X import Y" = lấy công cụ Y từ hộp dụng cụ X
from roboflow import Roboflow

# Dòng 2: Đăng nhập vào Roboflow bằng API key
# API key giống như "mật khẩu" để máy tính truy cập tài khoản
rf = Roboflow(api_key="8XQTfGjudP24VUAJUVkA")

# Dòng 3: Vào workspace → chọn project
# Workspace = "phòng làm việc" chứa nhiều project
# Project = 1 dự án cụ thể (ảnh sầu riêng đã gán nhãn)
project = rf.workspace("cico-siefo").project("durian-k51j3")

# Dòng 4: Chọn phiên bản dữ liệu
# Mỗi project có nhiều version (mỗi lần thêm/sửa ảnh là 1 version mới)
version = project.version(1)

# Dòng 5: Tải về định dạng YOLOv11
# "yolov11" = xuất nhãn theo format mà YOLO hiểu được
dataset = version.download("yolov11")
```

---

## 🧩 Giải thích chi tiết

### Roboflow là gì?

Roboflow là **website** giúp bạn:

1. **Upload ảnh** lá sầu riêng lên
2. **Gán nhãn** (annotate) — vẽ hình chữ nhật quanh vùng bệnh, đặt tên bệnh
3. **Xuất dữ liệu** về máy tính theo format phù hợp

```
Quy trình gán nhãn trên Roboflow:

  1. Upload ảnh         2. Vẽ bounding box      3. Gán tên bệnh
  ┌──────────┐        ┌──────────┐          ┌──────────┐
  │          │        │  ┌────┐  │          │  ┌────┐  │
  │   🍃    │  →→→  │  │    │  │  →→→     │  │BỆNH│  │
  │          │        │  └────┘  │          │  │CHÁY│  │
  │          │        │          │          │  │ LÁ │  │
  └──────────┘        └──────────┘          └──────────┘
```

### API key là gì?

```
API key = "chìa khóa" để máy tính tự đăng nhập vào Roboflow

Giống như:
  - Bạn đăng nhập Facebook bằng email + password (thủ công)
  - Code dùng API key để đăng nhập tự động (tự động)

⚠️ QUAN TRỌNG: API key phải giữ BÍ MẬT!
   Nếu bị lộ, người khác có thể truy cập dữ liệu của bạn.
```

### Dữ liệu tải về trông như thế nào?

Sau khi chạy `dataset = version.download("yolov11")`, bạn sẽ có thư mục:

```
durian-1/
├── train/                  ← Dùng để DẠY model (70-80% dữ liệu)
│   ├── images/             ← Thư mục chứa ảnh
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   └── labels/             ← Thư mục chứa nhãn (1 file .txt cho mỗi ảnh)
│       ├── img001.txt      ← Nhãn tương ứng với img001.jpg
│       ├── img002.txt
│       └── ...
│
├── valid/                  ← Dùng để KIỂM TRA trong lúc dạy (10-15%)
│   ├── images/
│   └── labels/
│
├── test/                   ← Dùng để THI CUỐI (10-15%)
│   ├── images/
│   └── labels/
│
└── data.yaml               ← File cấu hình (đường dẫn + tên các lớp)
```

### File nhãn (.txt) chứa gì?

Mỗi file `.txt` chứa thông tin bounding box, **mỗi dòng = 1 đối tượng**:

```
# Ví dụ nội dung file img001.txt:
2 0.45 0.62 0.15 0.20
12 0.78 0.31 0.10 0.08

# Giải thích dòng 1:
# 2      = class_id (bệnh cháy lá)
# 0.45   = tâm box nằm ở 45% chiều rộng ảnh
# 0.62   = tâm box nằm ở 62% chiều cao ảnh
# 0.15   = box rộng 15% chiều rộng ảnh
# 0.20   = box cao 20% chiều cao ảnh
```

```
  0%                    45%                   100%
  ┌────────────────────┬─────────────────────┐ 0%
  │                    │                     │
  │                    │                     │
  │                ┌───┼───┐                 │
  │                │   │   │                 │ 62%
  │                │   ●   │ ← tâm (0.45, 0.62)
  │                │       │                 │
  │                └───────┘                 │
  │                 ← 15% →                  │
  │                                          │
  └──────────────────────────────────────────┘ 100%
```

### data.yaml chứa gì?

```yaml
train: ./train/images    # Đường dẫn tới ảnh train
val: ./valid/images      # Đường dẫn tới ảnh validation
test: ./test/images      # Đường dẫn tới ảnh test

nc: 15                   # Số lượng classes (lớp)

names:                   # Tên từng class theo thứ tự ID
  0: benh bo tri
  1: benh phan trang
  2: benh-chay-la
  3: benh-dom-mat-cua
  ...
```

---

## ❓ Câu hỏi thường gặp

**Q: Tại sao tọa độ là 0-1 chứ không phải pixel?**  
A: Vì ảnh có nhiều kích cỡ khác nhau. Dùng tỷ lệ 0-1 giúp nhãn không phụ thuộc vào kích cỡ ảnh. Ví dụ: tâm tại 0.5, 0.5 luôn là **giữa ảnh** dù ảnh 640px hay 1920px.

**Q: Tại sao chia thành train/valid/test?**  
A:

- **Train**: Cho model "đọc sách" từ đây
- **Valid**: Cho model "thi thử" (kiểm tra trong lúc học)
- **Test**: "Thi thật" cuối cùng (chỉ dùng 1 lần sau khi train xong)

Nếu chỉ có train → model có thể "học thuộc lòng" mà không hiểu bài (overfitting).

---

> **Đọc tiếp:** [02_plot_class_distribution.md](./02_plot_class_distribution.md)
