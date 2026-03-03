# 🍈 Durian Leaf Disease Detection

Dự án Deep Learning phát hiện và phân loại bệnh trên lá cây sầu riêng sử dụng **YOLOv11x**.

## 📦 Cài đặt

```bash
# Clone project
git clone <repo-url>
cd durian-leaf-disease-detection

# Cài đặt dependencies (uv package manager)
uv sync
```

## ⚙️ Cấu hình

Toàn bộ cấu hình nằm trong `config/default.yaml`:

- **Roboflow** — API key, workspace, project
- **Paths** — đường dẫn dữ liệu, output
- **Class mapping** — ánh xạ class ID cũ → mới
- **Balance** — target classes, keep ratio
- **Training** — hyperparameters (batch, epochs, lr, ...)
- **Visualization** — splits, figsize, save options

> **Tip:** Thay vì sửa trực tiếp `default.yaml`, tạo file `config/custom.yaml` rồi dùng `--config config/custom.yaml`.

> **Bảo mật:** Đặt Roboflow API key qua biến môi trường: `export ROBOFLOW_API_KEY=your_key`

## 🚀 Sử dụng

### Chạy từng bước

```bash
# 1. Tải dữ liệu từ Roboflow
uv run main.py pull

# 2. Tái cấu trúc nhãn (gộp 15 lớp → 6 lớp)
uv run main.py refactor

# 3. Cân bằng dữ liệu (downsample lớp đa số)
uv run main.py balance

# 4. Vẽ biểu đồ phân phối lớp
uv run main.py plot

# 5. Huấn luyện model YOLOv11x
uv run main.py train
```

### Chạy toàn bộ pipeline

```bash
# Chạy tất cả 5 bước liên tiếp
uv run main.py pipeline

# Bỏ qua bước training (chỉ xử lý data)
uv run main.py pipeline --skip train_model

# Dùng config tùy chỉnh
uv run main.py pipeline --config config/custom.yaml

# Bật debug logging
uv run main.py pipeline -v
```

### Console script (sau khi install)

```bash
durian-detect pull
durian-detect train
durian-detect pipeline
```

## 📁 Cấu trúc thư mục

```
durian-leaf-disease-detection/
├── config/
│   └── default.yaml            # Cấu hình mặc định
├── src/
│   └── durian_detect/          # Package chính
│       ├── __init__.py
│       ├── cli.py              # CLI (argparse sub-commands)
│       ├── config.py           # Config loader (dataclass + YAML)
│       ├── data/
│       │   ├── pull.py         # Tải data từ Roboflow
│       │   ├── refactor.py     # Tái cấu trúc nhãn
│       │   └── balance.py      # Downsample classes
│       ├── visualization/
│       │   └── distribution.py # Plot class distribution
│       └── training/
│           └── train.py        # Huấn luyện YOLOv11x
├── scripts/
│   └── run_pipeline.py         # Script chạy full pipeline
├── main.py                     # Entry point
├── pyproject.toml              # Project metadata + dependencies
└── README.md
```

## 🏥 Các lớp bệnh

| ID  | Tên              | Mô tả                     |
| --- | ---------------- | ------------------------- |
| 0   | sau-an           | Sâu ăn                    |
| 1   | benh-chay-la     | Bệnh cháy lá              |
| 2   | benh-than-thu    | Bệnh thán thư             |
| 3   | benh-dom-mat-cua | Bệnh đốm mắt cua          |
| 4   | la-khoe          | Lá khỏe bình thường       |
| 5   | other            | Các bệnh khác (gộp chung) |

## 🛠️ Yêu cầu hệ thống

- Python ≥ 3.12
- GPU NVIDIA + CUDA (khuyến nghị cho training)
- uv package manager
