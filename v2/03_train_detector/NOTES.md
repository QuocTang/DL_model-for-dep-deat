# 03 — Train YOLO Detector

Train model **detect** (phát hiện vị trí + class) trên data đã refactor.

## File

### `train_yolo.py`

Smoke-test config (~15 phút mục tiêu, thực tế ~6 phút trên GTX 1650):

```python
model = YOLO("yolo11n.pt")        # nano, 2.6M params
model.train(
    data=str(DATA_YAML),          # → durian_refactor/data.yaml (6 class)
    project=str(RUNS_DIR),
    name=DEFAULT_RUN_NAME,        # "durian-1-yolo11n-smoke"
    epochs=8,
    imgsz=320,                    # half size = 4× faster
    batch=16,
    fraction=1.0,                 # dùng 100% data
    device=0 if cuda else "cpu",
    optimizer="AdamW",
    lr0=0.005, lrf=0.10,
    cos_lr=True,
    cache="ram",                  # 31GB RAM thoải mái
    plots=True,
    save_period=-1,               # chỉ save best.pt + last.pt
)
```

## Khái niệm

### YOLO = "You Only Look Once"
Object detection — **vẽ bounding box** + **gán class** cho từng vật trong ảnh. Khác classification (chỉ dán nhãn cho cả ảnh).

### Pretrained weights
`yolo11n.pt` đã được train sẵn trên dataset COCO (80 class generic). Khi `pretrained=True`, chỉ fine-tune lên data của bạn — nhanh hơn nhiều và tốt hơn so với train from scratch.

## Tham số quan trọng

| Tham số | Giá trị | Đánh đổi |
|---|---|---|
| `model = yolo11n` | nano (2.6M) | rất nhanh, accuracy thấp |
| `epochs=8` | 8 vòng | smoke test, không đủ cho production (cần 50-100) |
| `imgsz=320` | resize 320×320 | compute ¼ vs 640, mất chi tiết nhỏ |
| `batch=16` | 16 ảnh/bước | OK cho GTX 1650 4GB |
| `lr0=0.005` + `cos_lr` | LR đầu = 0.005, giảm cosine | đầu học nhanh, cuối tinh chỉnh |
| `cache="ram"` | load hết ảnh vào RAM | tăng tốc x2-3 |

## Metrics đọc thế nào

| Metric | Nghĩa |
|---|---|
| `box_loss` | Sai số tọa độ hộp dự đoán vs đúng (giảm = tốt) |
| `cls_loss` | Sai số class dự đoán (giảm = tốt) |
| `dfl_loss` | Distribution focal loss (chi tiết biên hộp) |
| **`P` (Precision)** | Trong các hộp model dự đoán, % đúng |
| **`R` (Recall)** | Trong các hộp đúng, % bắt được |
| **`mAP50`** | mean Average Precision với IoU≥0.5 — **CHÍNH** |
| **`mAP50-95`** | mAP trung bình qua nhiều ngưỡng IoU — KHẮC NGHIỆT HƠN |

### Trade-off P vs R
```
P cao, R thấp:  Đoán siêu chắc, bỏ sót nhiều  (overcautious)
P thấp, R cao:  Đoán bừa nhiều, bắt được nhiều (overeager)
P+R cân bằng:   ✅
```

mAP combine cả 2 → là số chính.

## Kết quả thực tế

```
       Ep1   Ep2   Ep3   Ep4   Ep5   Ep6   Ep7   Ep8
P      5%    64%   33%   22%   20%   28%   32%   32%
R      9%    14%   23%   38%   37%   40%   44%   43%
mAP50  2%    10%   16%   21%   21%   27%   30%   30%
```

**Final**: P=0.324, R=0.428, mAP50=**0.295**, mAP50-95=0.163

→ Vượt kỳ vọng baseline (mục tiêu >5% mAP50, đạt 30%).

## Cảnh báo trong log

### `AMP disabled trên GTX 1650`
GTX 1650 (Turing TU117) không có Tensor Core đầy đủ → AMP bị tắt. Train chậm hơn ~30% nhưng kết quả vẫn đúng.

### `1 duplicate labels removed`
1 file label có 2 dòng giống hệt → ultralytics tự dedupe. Noise nhỏ trong data Roboflow.

### `2 backgrounds`
2/230 ảnh valid không có annotation (có thể lá sạch hoặc bị lỗi label). Ultralytics dùng làm "negative example".

## Output (`v2/output/runs/detect/durian-1-yolo11n-smoke/`)

### Quan trọng
- **`weights/best.pt`** ⭐ — model có fitness cao nhất (epoch 7)
- `weights/last.pt` — epoch cuối
- `results.csv` + `results.png` — số/biểu đồ qua 8 epoch
- `confusion_matrix_normalized.png` — đường chéo càng đậm = càng đúng
- `BoxF1_curve.png`, `BoxPR_curve.png` — phân tích theo confidence threshold
- `val_batch*_pred.jpg` vs `val_batch*_labels.jpg` — so sánh trực quan

### Folder thừa
`durian-1-yolo11n-smoke-val/` được tạo do `model.val(...)` gọi explicit trong script. Subset của folder train, có thể bỏ.

## Phát hiện từ confusion matrix

Đọc theo cột (true class) — % model dự đoán:

| True | Đúng | Bỏ sót (background) | Ghi chú |
|---|:---:|:---:|---|
| sau-an | 40% | 41% | OK |
| benh-chay-la | 45% | 39% | Tốt nhất |
| benh-than-thu | 26% | 24% | Trung bình |
| benh-dom-mat-cua | **3%** | **91%** ⚠️ | Gần như không thấy |
| la-khoe-binh-thuong | **0%** | **93%** ⚠️ | Bỏ sót hoàn toàn |
| others | 30% | 69% | Lẫn lộn |

→ Confirm dự đoán EDA: **class hiếm bị bỏ sót**. Cần `class_weights` trong run thật.

## Lệnh chạy
```bash
cd v2
uv run 03_train_detector/train_yolo.py
explorer.exe output/runs/detect/durian-1-yolo11n-smoke/
```

## Cho run thật (sau này, ~vài giờ)

```python
model = YOLO("yolo11s.pt")  # hoặc yolo11m
model.train(
    data=str(DATA_YAML),
    epochs=50,                  # đủ hội tụ
    imgsz=640,                  # full resolution
    batch=8,
    # class_weights=...         # cân bằng class hiếm
    patience=15,                # early-stop
)
```
Mục tiêu mAP50 ~ 0.6-0.8.
