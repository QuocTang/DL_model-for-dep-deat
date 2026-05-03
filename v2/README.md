# Durian Disease Detection — v2

Pipeline phát hiện bệnh trên lá sầu riêng: **YOLOv11 detector** trích đặc trưng sâu, sau đó **classical ML** (XGBoost / RandomForest / LightGBM) phân loại.

## Cấu trúc

```
v2/
├── config.py              ← TẤT CẢ đường dẫn (data, output) khai báo ở đây
├── pyproject.toml         ← deps (uv-managed)
├── data/                  ← INPUT
│   ├── durian-1/          ← raw dataset (sau khi pull)
│   └── durian_refactor/   ← sau khi refactor labels
├── output/                ← OUTPUT (mọi file sinh ra)
│   ├── runs/detect/       ← YOLO training
│   ├── extracted_features/
│   ├── extracted_features_top6/
│   ├── ml_models/
│   └── model_comparison/
│
├── 01_data/               1. Lấy & xử lý dữ liệu
├── 02_eda/                2. EDA – xem phân bố nhãn
├── 03_train_detector/     3. Train YOLO (sinh best.pt)
├── 04_features/           4. Trích đặc trưng deep từ YOLO
├── 05_train_classifier/   5. Train ML head + so sánh
└── 06_inference/          6. Inference (CLI + UI)
```

## Flow đầy đủ (chạy lần lượt)

> Mọi lệnh chạy từ thư mục `v2/` (`cd v2` trước), dùng `uv run <file>` thay cho `python <file>`.

| Bước | Lệnh | Output |
|---|---|---|
| 1.1 Pull dataset | `uv run 01_data/pull_data.py` | `data/durian-1/` |
| 1.2 (Tùy chọn) Cân bằng class lớn | `uv run 01_data/downsample_large_classes.py` | `data/durian-1_balanced/` |
| 1.3 Refactor nhãn (15 → 6 class) | `uv run 01_data/refactor_data.py` | `data/durian_refactor/` |
| 2. EDA phân bố | `uv run 02_eda/plot_class_distribution.py` | biểu đồ |
| 3. Train YOLO | `uv run 03_train_detector/train_yolo.py` | `output/runs/detect/durian-1-yolo11x/weights/best.pt` |
| 4.1 Trích feature | `uv run 04_features/extract_deep_features.py` | `output/extracted_features/{train,valid,test}_features.csv` |
| 4.2 Lọc top-K class | `uv run 04_features/filter_top_classes.py` | `output/extracted_features_top6/` |
| 5.1 Train ML | `uv run 05_train_classifier/train_ml_features.py --model xgb` | `output/ml_models/xgb_model.joblib` |
| 5.2 So sánh model | `uv run 05_train_classifier/compare_models.py --comparison-dir output/model_comparison` | CSV + Markdown |
| 6.a Inference CLI | `uv run 06_inference/cli.py --image <path>` | JSON kết quả |
| 6.b UI Streamlit | `uv run streamlit run 06_inference/ui_streamlit.py` | web UI |
| 6.c UI PySide6 | `uv run 06_inference/ui_pyside.py` | desktop UI |

### ⚠️ Lưu ý thứ tự bước 1

`downsample_large_classes.py` dùng **class ID gốc** (2, 12 = benh-chay-la, sau-an), nên **phải chạy TRƯỚC `refactor_data.py`** (refactor đổi class ID). Nếu chạy ngược, downsample không tìm được class để bỏ.

### 💡 Cho người mới: lần đầu nên skip 1.2

Bỏ qua `downsample_large_classes.py` cho lần chạy đầu vì:
- Mất 60% ảnh class lớn — có thể làm model yếu hơn
- Chỉ xử lý split `train`, không đụng valid/test → metric không phản ánh đúng
- Output `durian-1_balanced/` không kết nối với `durian_refactor/` (refactor không đọc folder này)
- Cách balance tốt hơn: dùng `class_weights` trong `model.train()` của ultralytics — sẽ làm sau khi có baseline metrics

## Cài đặt

```bash
cd v2
uv sync          # hoặc: pip install -e .
```

Yêu cầu Python >= 3.13. Đặt API key Roboflow trước khi chạy bước 1.1 (xem `pull_data.py`).

## Ghi chú thiết kế

- **Mọi đường dẫn quy về `config.py`** — không hard-code paths trong script.
- **Outputs gộp vào `v2/output/`** — dễ xóa/zip/share.
- **Folder số thứ tự** chỉ để đọc từ trên xuống. Không phải Python package; các script tự `sys.path.insert(parent)` để import `config`.
