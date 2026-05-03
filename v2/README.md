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

| Bước | Lệnh | Output |
|---|---|---|
| 1.1 Pull dataset | `python v2/01_data/pull_data.py` | `v2/data/durian-1/` |
| 1.2 Refactor nhãn | `python v2/01_data/refactor_data.py` | `v2/data/durian_refactor/` |
| 1.3 (Tùy chọn) Cân bằng | `python v2/01_data/downsample_large_classes.py` | `v2/data/durian-1_balanced/` |
| 2. EDA phân bố | `python v2/02_eda/plot_class_distribution.py` | biểu đồ |
| 3. Train YOLO | `python v2/03_train_detector/train_yolo.py` | `v2/output/runs/detect/durian-1-yolo11x/weights/best.pt` |
| 4.1 Trích feature | `python v2/04_features/extract_deep_features.py` | `v2/output/extracted_features/{train,valid,test}_features.csv` |
| 4.2 Lọc top-K class | `python v2/04_features/filter_top_classes.py` | `v2/output/extracted_features_top6/` |
| 5.1 Train ML | `python v2/05_train_classifier/train_ml_features.py --model xgb` | `v2/output/ml_models/xgb_model.joblib` |
| 5.2 So sánh | `python v2/05_train_classifier/compare_models.py --comparison-dir v2/output/model_comparison` | CSV + Markdown |
| 6.a Inference CLI | `python v2/06_inference/cli.py --image <path>` | JSON kết quả |
| 6.b UI Streamlit | `streamlit run v2/06_inference/ui_streamlit.py` | web UI |
| 6.c UI PySide6 | `python v2/06_inference/ui_pyside.py` | desktop UI |

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
