# 04 — Deep Feature Extraction (cho người mới)

> Mục tiêu: Đọc xong file này, **không cần search ở đâu khác** vẫn hiểu Bước 4 làm gì, tại sao cần, và mỗi đoạn code đang làm gì.

---

## 1. Tại sao cần Bước 4?

### Câu chuyện
- **Bước 3** đã train xong YOLO → model biết "đây là lá bị sâu ăn", "đây là chấm mắt cua"…
- Nhưng YOLO làm **detection** (vẽ hộp + gán nhãn cho từng vật trong ảnh).
- Mình lại muốn **classification** (1 nhãn cho cả ảnh) bằng **XGBoost** (một thuật toán ML cổ điển — nhanh, dễ giải thích).
- Vấn đề: XGBoost không "nhìn" được ảnh thô. Nó chỉ ăn **vector số** (ví dụ `[0.12, -0.04, 1.3, ...]`).

→ **Bước 4 = cây cầu**: Biến mỗi ảnh thành 1 vector số (512 chiều) bằng cách "mượn não" của YOLO đã train.

### Analogy
Hãy tưởng tượng YOLO là một **nhà phê bình tranh** đã xem hàng ngàn bức tranh.
- Bước 3: dạy nó "đây là tranh Picasso, đây là Van Gogh".
- Bước 4: bảo nó "với mỗi bức tranh, hãy mô tả thành 512 con số đặc trưng nhất".
- Bước 5: đưa 512 số đó cho 1 học sinh khác (XGBoost) học "512 số này → tranh ai".

Tại sao không để YOLO làm hết? Vì:
- YOLO mạnh ở **vị trí object**, yếu ở **classify cả ảnh**.
- XGBoost rất mạnh ở classification trên vector. Combo = win.

---

## 2. Khái niệm cơ bản (đọc 1 lần là đủ)

### 2.1. Mạng neural có nhiều "tầng" (layer)
Một CNN (Convolutional Neural Network) như YOLO gồm nhiều layer xếp chồng:

```
Ảnh đầu vào (3 × 320 × 320)
   ↓ layer 0  → 16 ảnh nhỏ (16 × 160 × 160)
   ↓ layer 1  → 32 ảnh nhỏ (32 × 80  × 80 )
   ↓ layer 2  → 64 ảnh nhỏ (64 × 40  × 40 )
   ...
   ↓ layer 22 → kết quả detection (hộp + class)
```

Mỗi layer **biến đổi** input từ layer trước. Layer càng sâu, thông tin càng "trừu tượng":
| Layer | Học cái gì? |
|---|---|
| Đầu (0-3) | Cạnh, góc, màu — đặc trưng pixel |
| Giữa (4-9) | Texture (vân lá), hình dạng nhỏ (đốm, gân) |
| Cuối (10-22) | Vật thể hoàn chỉnh (cả lá, cả vết bệnh) |

### 2.2. Backbone vs Head
YOLO chia làm 2 phần:
- **Backbone** = phần "nhìn" ảnh (layer 0 → 9). Phần này **tổng quát**, dùng được cho nhiều bài toán.
- **Head** = phần ra kết quả detection (layer 10 → 22). Phần này **chuyên biệt** cho task detect.

→ Mình **chỉ lấy từ backbone** vì nó general, dùng cho classification được.

### 2.3. Feature map (= activation) là gì?
Output của 1 layer = một **stack các "ảnh xám" nhỏ**.

Ví dụ layer 4 của YOLO11n cho ra: `[64 channels × 40 × 40]`
- 64 = số "kênh" (mỗi kênh phát hiện 1 loại pattern: kênh 1 phát hiện cạnh dọc, kênh 2 phát hiện vết tròn…)
- 40 × 40 = bản đồ vị trí (mỗi ô tương ứng 1 vùng 8×8 pixel của ảnh gốc)

Có thể xem đây là **64 tấm bản đồ nhiệt** — mỗi tấm đánh dấu "vùng nào của ảnh kích hoạt pattern này".

### 2.4. Pooling 1×1 (trung bình toàn ảnh)
Mình không cần biết "vết bệnh ở góc nào", chỉ cần "ảnh này có vết bệnh không". → Lấy **trung bình** cả tấm bản đồ 40×40 thành **1 số duy nhất**.

```
Tấm bản đồ 40×40 với 1600 ô số  ─AvgPool─►  1 số (= trung bình 1600 ô)
```

Áp dụng cho 64 channels → **64 số** (1 số/channel).

### 2.5. Hook là gì?
Bình thường khi gọi `model(image)`, YOLO chạy hết từ đầu đến cuối, chỉ trả về **kết quả cuối** (detection).

Mình muốn **chộp output ở giữa** (layer 4, 6, 9). Cách: gắn 1 "hook" — như **đặt camera** ở giữa pipeline để ghi lại output mà không phá code gốc của YOLO.

```python
# Pseudocode
def hook(layer_output):
    save_to_buffer(layer_output)  # mình chộp ở đây

model.layer[4].register_hook(hook)
model(image)  # YOLO chạy bình thường, hook tự động chộp
```

---

## 3. Pipeline Bước 4 — chạy 1 ảnh đi qua từng giai đoạn

```
┌─────────────────────────────────────────────────────────────┐
│  Ảnh gốc (3 × 320 × 320)                                    │
└──────────────────────────┬──────────────────────────────────┘
                           ↓ (đẩy vào YOLO best.pt từ Bước 3)
┌─────────────────────────────────────────────────────────────┐
│  YOLO Backbone chạy forward                                 │
│  ┌─Layer 4─┐    ┌─Layer 6─┐    ┌─Layer 9─┐                  │
│  │ chộp 📷 │    │ chộp 📷 │    │ chộp 📷 │  ← HOOK chộp     │
│  │64×40×40 │    │128×20×20│    │320×10×10│                  │
│  └────┬────┘    └────┬────┘    └────┬────┘                  │
└───────┼──────────────┼──────────────┼───────────────────────┘
        ↓              ↓              ↓
   AvgPool 1×1    AvgPool 1×1    AvgPool 1×1   ← trung bình toàn map
        ↓              ↓              ↓
    64 số          128 số         320 số
        └──────────────┼──────────────┘
                       ↓ Concat (nối thẳng)
              ╔═══════════════════╗
              ║  Vector 512 chiều ║   ← đây là feature
              ╚═══════════════════╝
                       ↓
              Lưu vào CSV cùng với label
```

### Vector cuối cùng
```
[0.12, -0.04, 1.3, 0.0, 2.1, ..., -0.5]   ← 512 số
 └────64 từ layer 4────┘└──128 từ layer 6──┘└──320 từ layer 9──┘
```

(Đoán ban đầu là 64+128+256=448, nhưng thực tế YOLO11n có channels [64, 128, 320] = 512.)

---

## 4. Tại sao chọn layer **4, 6, 9**?

| Layer | Thông tin | Nên lấy? |
|---|---|---|
| 0-3 | Cạnh, màu thô | ❌ Quá thấp, mọi ảnh giống nhau |
| **4-9** | **Texture, hình nhỏ** | ✅ Phong phú, phân biệt được bệnh |
| 10+ | Object hoàn chỉnh | ❌ Quá chuyên biệt cho detection task |

Lấy 3 layer (không phải 1) để có **đa độ phân giải** — vừa thấy chi tiết nhỏ vừa thấy hình tổng quan.

---

## 5. Một ảnh = một vector + một nhãn (dominant)

YOLO label là **detection** (1 ảnh có thể nhiều object, nhiều class). Nhưng classification cần **1 nhãn/ảnh**.

Cách giải quyết: lấy **class xuất hiện NHIỀU NHẤT** trong các annotation của ảnh đó.

```python
# Một file label .txt có 5 dòng:
# 0  ...  ← sau-an
# 0  ...  ← sau-an
# 0  ...  ← sau-an
# 1  ...  ← benh-chay-la
# 3  ...  ← benh-dom-mat-cua

class_ids = [0, 0, 0, 1, 3]
dominant_class = max(set(class_ids), key=class_ids.count)  # → 0 (sau-an)
```

→ Ảnh này gán nhãn `sau-an`.

---

## 6. ⚠️ Lưu ý quan trọng: imgsz phải KHỚP với train

Bước 3 train YOLO ở `imgsz=320` → Bước 4 extract cũng phải **320**.

Vì sao? Vì model học cách "nhìn" ảnh ở size đó. Đưa size khác (vd 640) vào → activation của các layer sẽ khác → vector không còn "đúng nghĩa".

```bash
uv run 04_features/extract_deep_features.py --imgsz 320   # ✅ ĐÚNG
uv run 04_features/extract_deep_features.py --imgsz 640   # ❌ SAI
```

---

## 7. Output (`v2/output/extracted_features/`)

3 file CSV (mỗi split 1 file):

| File | Số ảnh | Số cột |
|---|---:|---:|
| `train_features.csv` | 2359 | 515 |
| `valid_features.csv` |  227 | 515 |
| `test_features.csv`  |  112 | 515 |

**515 cột = 512 features + label + class_name + image_name**

Mỗi dòng:
```
feat_0, feat_1, ..., feat_511, label, class_name,           image_name
0.12,   -0.04,  ..., -0.5,     0,     sau-an,               IMG_001.jpg
```

44/2403 ảnh train bị skip (file label rỗng hoặc không tồn tại) — bình thường.

---

## 8. 🎯 Phát hiện thú vị: Imbalance giảm mạnh

Vì giờ phân loại theo **ảnh** (1 nhãn/ảnh) thay vì theo **annotation** (nhiều/ảnh), class lớn (chay-la, sau-an) bị "gộp":

| Class | EDA (annotations) | Features (images) | Tỉ lệ giảm |
|---|---:|---:|---:|
| benh-chay-la | 4039 | 778 | ÷5.2 |
| sau-an | 3399 | 639 | ÷5.3 |
| benh-than-thu | 596 | 315 | ÷1.9 |
| benh-dom-mat-cua | 1113 | 222 | ÷5.0 |
| others | 605 | 204 | ÷3.0 |
| la-khoe-binh-thuong | 228 | 201 | ÷1.1 |

| | Imbalance ratio | Tác động |
|---|---:|---|
| EDA (annotations) | 18× | YOLO khó học class hiếm |
| Features (images) | **3.9×** | XGBoost dễ học hơn nhiều |

→ Bài toán classification ở Bước 5 sẽ **DỄ hơn** YOLO detection.

---

## 9. Lệnh chạy

```bash
cd v2
uv run 04_features/extract_deep_features.py --imgsz 320
```

Tham số khác (không bắt buộc):
- `--model <path>` — đổi model best.pt khác
- `--data-yaml <path>` — đổi dataset
- `--layers 4,6,9` — đổi layer nào lấy feature (nếu muốn thử nghiệm)
- `--max-images N` — limit để test nhanh

---

## 10. Sau Bước 4 thì đi đâu?

→ **Bước 5**: dùng 3 file CSV này train **XGBoost classifier**.
- Input: 512 cột feature
- Output: dự đoán 1 trong 6 class
- Đánh giá: accuracy, F1 trên `test_features.csv`

File: `v2/05_train_classifier/train_ml_features.py`

---

## 11. (Tùy chọn) `filter_top_classes.py` — lọc top-K class

### Mục đích
Sau khi train ML lần đầu (Bước 5), nếu có class quá yếu (F1 thấp / data quá ít), có thể **vứt bớt** rồi train lại để model "tập trung" vào class còn lại.

### Khi nào CẦN chạy?
| Tình huống | Có cần? |
|---|---|
| Dataset nhiều class (vd 15+) và 1 vài class rất yếu | ✅ Nên |
| Đã refactor 15→6 ở Bước 1 và muốn giữ cả 6 | ❌ KHÔNG cần |
| Muốn thử model "chỉ 4 class mạnh nhất" để demo | ✅ OK |

→ **Hiện tại không cần** vì đã refactor xuống 6 class ở Bước 1 rồi.

### Cách hoạt động
```
Input:
  v2/output/extracted_features/{train,valid,test}_features.csv  (6 class)
  v2/output/ml_models/classification_report_valid.csv           (F1 mỗi class)

  ↓ Rank class theo (valid_F1 desc, train_count desc)
  ↓ Bỏ class có train_count < 30 (data quá ít)
  ↓ Giữ top-K (default 6)
  ↓ Remap label cũ → label mới liên tục (0..K-1)

Output:
  v2/output/extracted_features_top6/{train,valid,test}_features.csv
  v2/output/extracted_features_top6/selection_summary.json
```

### 2 chế độ

**Chế độ AUTO** (mặc định) — rank theo F1:
```bash
uv run 04_features/filter_top_classes.py --top-k 4
```
→ Giữ 4 class có F1 cao nhất + đủ data.

**Chế độ MANUAL** — tự chỉ định:
```bash
uv run 04_features/filter_top_classes.py \
    --class-list "sau-an,benh-chay-la,la-khoe-binh-thuong,benh-dom-mat-cua"
```

### Lưu ý "remap label"
Sau khi bỏ class, label cũ có thể `[0, 1, 3, 4]` (thiếu 2). XGBoost cần label **liên tục** `[0, 1, 2, 3]`. Script tự đổi:
```
label cũ:  0  1  3  4
label mới: 0  1  2  3
```
Cột `label_raw` giữ giá trị gốc để debug.

### Pipeline nếu dùng filter
```
Bước 4a: extract_deep_features.py    → 6-class CSV
Bước 5a: train_ml_features.py        → ra report
Bước 4b: filter_top_classes.py       → 4-class CSV  (nếu muốn lọc)
Bước 5b: train_ml_features.py \
            --features-dir output/extracted_features_top6 \
            --output-dir output/ml_models_top4
```

→ Train **2 lần**: lần 1 để biết class nào yếu, lần 2 sau khi lọc.

---

## Tóm tắt 1 dòng

> Bước 4 = bóp não YOLO ra **512 con số** cho mỗi ảnh, để Bước 5 (XGBoost) học classification. `filter_top_classes.py` là tùy chọn — chỉ dùng nếu muốn vứt bớt class yếu sau khi đã train xong lần 1.
