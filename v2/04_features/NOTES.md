# 04 — Deep Feature Extraction

Dùng YOLO `best.pt` (đã train ở Bước 3) để **biến mỗi ảnh thành 1 vector số**, làm input cho ML cổ điển ở Bước 5.

## Files

### `extract_deep_features.py`
Lấy activation ở các **layer giữa** của YOLO backbone, pooling thành vector.

### `filter_top_classes.py`
(Tùy chọn, dùng sau khi có classification report) — lọc giữ lại top-K class theo F1 + train support, remap label thành liên tục.

## Khái niệm: Deep Features

Khi YOLO xử lý ảnh, mỗi layer ngày càng "trừu tượng":
- Layer đầu: cạnh, màu
- Layer giữa: texture, hình nhỏ
- Layer cuối: vật thể

Lấy activation ở layer giữa = "cách YOLO nhìn ảnh dưới dạng vector" — thông tin phong phú hơn pixel thô. Truyền vào XGBoost dễ học hơn nhiều.

```
Ảnh → YOLO11n → HOOK ở [layer 4, 6, 9] → Adaptive Pool 1×1 → Concat → Vector 512-D
```

## Tham số

```python
DEFAULT_LAYERS = [4, 6, 9]  # YOLO11n có ~23 layer
```

Mỗi layer cho 1 vector size = số channel. Pool 1×1 → trung bình toàn ảnh thành 1 số/channel. Concat 3 layer → vector cuối cùng.

```python
final_vector = pool(layer_4) ⊕ pool(layer_6) ⊕ pool(layer_9)
            =  ?            +  ?            +  ?            = 512
```

(Dự đoán trước khi chạy: 64+128+256=448. Thực tế: 512 — channel sizes của yolo11n hơi khác mình tưởng.)

## ⚠️ Lưu ý quan trọng

### Phải dùng cùng `imgsz` với train
Train YOLO ở `imgsz=320` → **extract cũng phải 320**. Nếu khác, features sẽ không khớp với cách model học.

```bash
uv run 04_features/extract_deep_features.py --imgsz 320
```

### Một ảnh = một vector + một nhãn (dominant)
Khác với detection (có thể nhiều object/ảnh), classification chỉ 1 nhãn/ảnh:
```python
class_ids = [int(line.split()[0]) for line in labels]
dominant_class = max(set(class_ids), key=class_ids.count)
```
→ Lấy class **xuất hiện nhiều nhất** trong các annotation của ảnh đó.

## Output (`v2/output/extracted_features/`)

```
train_features.csv  →  2359 ảnh × 515 cột (512 features + label + class_name + image_name)
valid_features.csv  →   227 ảnh × 515 cột
test_features.csv   →   112 ảnh × 515 cột
```

44/2403 train images skip (không có label hoặc label rỗng) — bình thường.

## 🎯 Phát hiện thú vị: imbalance giảm mạnh

Vì phân loại theo **ảnh** (không phải annotation), class lớn (chay-la, sau-an thường nhiều annotation/ảnh) bị "gộp" lại:

| Class | EDA (annotations) | Features (images) | Tỉ lệ giảm |
|---|---:|---:|---:|
| benh-chay-la | 4039 | 778 | ÷5.2 |
| sau-an | 3399 | 639 | ÷5.3 |
| benh-than-thu | 596 | 315 | ÷1.9 |
| benh-dom-mat-cua | 1113 | 222 | ÷5.0 |
| others | 605 | 204 | ÷3.0 |
| la-khoe-binh-thuong | 228 | 201 | ÷1.1 |

| | Imbalance | Tác động |
|---|---:|---|
| EDA (annotations) | 18× | YOLO khó học class hiếm |
| Features (images) | **3.9×** | XGBoost dễ học hơn nhiều |

→ Bài toán classification ở Bước 5 sẽ DỄ hơn YOLO detection rất nhiều.

## Lệnh chạy

```bash
cd v2
uv run 04_features/extract_deep_features.py --imgsz 320
```

Tham số khác:
- `--model <path>` — đổi best.pt khác
- `--data-yaml <path>` — đổi dataset
- `--layers 4,6,9` — đổi layer nào lấy feature
- `--max-images N` — limit để test nhanh
